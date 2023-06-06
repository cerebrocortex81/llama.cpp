#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK_K 256

typedef struct {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_k;

static void dequantize_row_q4_0(device const block_q4_0 * x, device float * y, int k) {
    const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

static inline uchar2 get_scale_min_k4(int j, device const uint8_t * q) {
    uchar2 r;
    if (j < 4) {
        r[0] = q[j] & 63; r[1] = q[j + 4] & 63;
    } else {
        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
    return r;
}

static void dequantize_row_q4_k(device const block_q4_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = x[i].d;
        const float min = x[i].dmin;

        device const uint8_t * q = x[i].qs;
        device const uint8_t * scales = x[i].scales;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            const uchar2 sc1 = get_scale_min_k4(is, scales);
            const float d1 = d * sc1[0]; const float m1 = min * sc1[1];
            const uchar2 sc2 = get_scale_min_k4(is+1, scales);
            const float d2 = d * sc2[0]; const float m2 = min * sc2[1];
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }

    }
}

kernel void kernel_add(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

kernel void kernel_mul(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % ne00];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        threadgroup float  * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    buf[tpitg[0]] = -INFINITY;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] = MAX(buf[tpitg[0]], psrc0[i00]);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] = MAX(buf[tpitg[0]], buf[tpitg[0] + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float max = buf[0];

    // parallel sum
    buf[tpitg[0]] = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] += exp(psrc0[i00] - max);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] += buf[tpitg[0] + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sum = buf[0];

    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        pdst[i00] = exp(psrc0[i00] - max) / sum;
    }
}

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_get_rows_f16(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    for (int j = 0; j < ne00; j++) {
        dst[i*nb1 + j] = ((device half *) ((device char *) src0 + r*nb01))[j];
    }
}

kernel void kernel_get_rows_q4_0(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_0(
            (device const block_q4_0 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q4_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_k(
            (device const block_q4_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);

    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00] * x[i00];
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    device float * y = dst + tgpig*ne00;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_mul_mat_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2  tpig[[thread_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {
    const int nb = ne00/QK4_0;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q4_0 * x = (device const block_q4_0 *) src0 + r0*nb;
    device const float      * y = (device const float      *) src1 + r1*ne10;

    const uint nth = tptg.x*tptg.y;
    const uint ith = tptg.y*tpitg.x + tpitg.y;

    sum[ith] = 0.0f;

    for (int i = tpitg.x; i < nb; i += tptg.x) {
        device const uchar4 * x0p = (device const uchar4 *) (x + i)->qs;
        device const float4 * y0p = (device const float4 *) (y + i*QK4_0);

        const float d = (float)((x + i)->d);

        const uchar4 x0v = *(x0p + tpitg.y);
        const float4 y0v = *(y0p + tpitg.y + 0);
        const float4 y1v = *(y0p + tpitg.y + 4);

        float acc = 0.0f;

        for (int j = 0; j < 4; ++j) {
            const int x0 = x0v[j] & 0x0F;
            const int x1 = x0v[j] >>   4;

            const float y0 = y0v[j];
            const float y1 = y1v[j];

            acc += (x0 - 8)*y0 + (x1 - 8)*y1;
        }

        sum[ith] += acc*d;
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = nth/2; i > 0; i /= 2) {
        if (ith < i) {
            sum[ith] += sum[ith + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ith == 0) {
        dst[r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_mul_mat_q4_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2  tpig[[thread_position_in_grid]],               // we don't use this for now
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q4_k * x = (device const block_q4_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const uint nth = tptg.x*tptg.y;
    const uint ith = tptg.y*tpitg.x + tpitg.y;

    const int tid = tpitg.y;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int n   = 4;
    const int is  = 2*il;

    sum[ith] = 0.0f;

    float sumf = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        device const uint8_t * q = (x + i)->qs + 32*il + n*ir;
        device const float   * y = yy + i*QK_K + 64*il + n*ir;
        device const uint8_t * scales = (x + i)->scales;

        const float dall = (float)((x + i)->d);
        const float dmin = (float)((x + i)->dmin);

        const uchar2 sc1 = get_scale_min_k4(is, scales);
        const float d1 = dall * sc1[0]; const float m1 = dmin * sc1[1];
        const uchar2 sc2 = get_scale_min_k4(is+1, scales);
        const float d2 = dall * sc2[0]; const float m2 = dmin * sc2[1];

        for (int l = 0; l < n; ++l) {
            sumf += y[l] * (d1 * (q[l] & 0xF) - m1) + y[l+32] * (d2 * (q[l] >> 4) - m2);
        }

    }
    sum[ith] = sumf;

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = nth/2; i > 0; i /= 2) {
        if (ith < i) {
            sum[ith] += sum[ith + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (ith == 0) {
        dst[r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_mul_mat_f16_f32(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tpig[[thread_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    device const half  * x = (device const half  *) (src0 + r0*nb01 + im*nb02);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    sum[tpitg.x] = 0.0f;

    for (int i = tpitg.x; i < ne00; i += tptg.x) {
        sum[tpitg.x] += (float) x[i] * (float) y[i];
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tptg.x/2; i > 0; i /= 2) {
        if (tpitg.x < i) {
            sum[tpitg.x] += sum[tpitg.x + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tpitg.x == 0) {
        dst[im*ne1*ne0 + r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_rope(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant       int & n_past,
        constant       int & n_dims,
        constant       int & mode,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i3 = tpig[2];
    const int64_t i2 = tpig[1];
    const int64_t i1 = tpig[0];

    const bool is_neox = mode & 2;
    const float theta_scale = pow(10000.0, -2.0f/n_dims);

    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);

    float theta = (float)p;

    if (!is_neox) {
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cos(theta);
            const float sin_theta = sin(theta);

            theta *= theta_scale;

            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        // TODO: implement
    }
}

kernel void kernel_cpy_f32_f16(
        device const float * src0,
        device        half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}
