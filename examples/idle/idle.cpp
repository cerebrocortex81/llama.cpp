#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void print_usage(int /*argc*/, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-ngl n_gpu_layers]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;

    // number of layers to offload to the GPU
    int ngl = 99;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // we need just a dummy token to evaluate
    std::vector<llama_token> prompt_tokens(1, llama_token_bos(model));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = 512;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    const int n_iters = 10;

    // warm-up
    llama_decode(ctx, batch);
    llama_kv_cache_clear (ctx);
    llama_kv_cache_update(ctx);
    llama_synchronize    (ctx);

    for (int64_t t_pause_ms = 0; t_pause_ms <= 2200; t_pause_ms += 200) {
        double t_sum_us  = 0.0;
        double t_sum2_us = 0.0;

        for (int i = 0; i < n_iters; i++) {
            // this pause is important - it simulates "idle GPU"
            std::this_thread::sleep_for(std::chrono::milliseconds(t_pause_ms));

            const int64_t t_start_us = llama_time_us();

            // this should take constant time
            llama_decode(ctx, batch);
            llama_synchronize(ctx);

            const int64_t t_end_us = llama_time_us();

            const double t_cur_us = t_end_us - t_start_us;

#if 0
            // print individual decode times
            printf("  - decode time: %8.2f ms\n", t_cur_us / 1000);
#endif

            t_sum_us  += t_cur_us;
            t_sum2_us += t_cur_us * t_cur_us;

            llama_kv_cache_clear (ctx);
            llama_kv_cache_update(ctx);
            llama_synchronize    (ctx); // just in case
        }

        const double t_avg_us = t_sum_us / n_iters;
        const double t_dev_us = sqrt((t_sum2_us / (n_iters - 1)) - (t_avg_us * t_avg_us * n_iters) / (n_iters - 1));

        printf("iters: %4d, pause: %5d ms, avg decode time: %8.2f +/- %4.2f ms\n", n_iters, (int) t_pause_ms, t_avg_us / 1000, t_dev_us / 1000);
        fflush(stdout);
    }

    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
