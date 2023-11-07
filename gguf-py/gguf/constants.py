from __future__ import annotations

import sys
from enum import Enum, IntEnum, auto
from typing import Any, NamedTuple

#
# constants
#

GGUF_MAGIC             = 0x46554747 # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32

#
# metadata keys
#

class GeneralKeys(NamedTuple):
    ARCHITECTURE         = "general.architecture"
    QUANTIZATION_VERSION = "general.quantization_version"
    ALIGNMENT            = "general.alignment"
    NAME                 = "general.name"
    AUTHOR               = "general.author"
    URL                  = "general.url"
    DESCRIPTION          = "general.description"
    LICENSE              = "general.license"
    SOURCE_URL           = "general.source.url"
    SOURCE_HF_REPO       = "general.source.huggingface.repository"
    FILE_TYPE            = "general.file_type"

class AttentionKeys(NamedTuple):
    HEAD_COUNT        = "{arch}.attention.head_count"
    HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
    MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
    CLAMP_KQV         = "{arch}.attention.clamp_kqv"
    LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
    LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"

class RopeKeys(NamedTuple):
    DIMENSION_COUNT         = "{arch}.rope.dimension_count"
    FREQ_BASE               = "{arch}.rope.freq_base"
    SCALING_TYPE            = "{arch}.rope.scaling.type"
    SCALING_FACTOR          = "{arch}.rope.scaling.factor"
    SCALING_ORIG_CTX_LEN    = "{arch}.rope.scaling.original_context_length"
    SCALING_FINETUNED       = "{arch}.rope.scaling.finetuned"

class TokenizerKeys(NamedTuple):
    MODEL      = "tokenizer.ggml.model"
    LIST       = "tokenizer.ggml.tokens"
    TOKEN_TYPE = "tokenizer.ggml.token_type"
    SCORES     = "tokenizer.ggml.scores"
    MERGES     = "tokenizer.ggml.merges"
    BOS_ID     = "tokenizer.ggml.bos_token_id"
    EOS_ID     = "tokenizer.ggml.eos_token_id"
    UNK_ID     = "tokenizer.ggml.unknown_token_id"
    SEP_ID     = "tokenizer.ggml.seperator_token_id"
    PAD_ID     = "tokenizer.ggml.padding_token_id"
    HF_JSON    = "tokenizer.huggingface.json"
    RWKV       = "tokenizer.rwkv.world"

class LLMKeys(NamedTuple):
    CONTEXT_LENGTH        = "{arch}.context_length"
    EMBEDDING_LENGTH      = "{arch}.embedding_length"
    BLOCK_COUNT           = "{arch}.block_count"
    FEED_FORWARD_LENGTH   = "{arch}.feed_forward_length"
    USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
    TENSOR_DATA_LAYOUT    = "{arch}.tensor_data_layout"

class Keys(NamedTuple):
    GENERAL   = GeneralKeys()
    LLM       = LLMKeys()
    ATTENTION = AttentionKeys()
    ROPE      = RopeKeys()
    TOKENIZER = TokenizerKeys()

KEY = Keys()

#
# recommended mapping of model tensor names for storage in gguf
#


class MODEL_ARCH(IntEnum):
    LLAMA         : int = auto()
    FALCON        : int = auto()
    BAICHUAN      : int = auto()
    GPT2          : int = auto()
    GPTJ          : int = auto()
    GPTNEOX       : int = auto()
    MPT           : int = auto()
    STARCODER     : int = auto()
    PERSIMMON     : int = auto()
    REFACT        : int = auto()
    BERT          : int = auto()
    BLOOM         : int = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD      : int = auto()
    TOKEN_EMBD_NORM : int = auto()
    TOKEN_TYPES     : int = auto()
    POS_EMBD        : int = auto()
    OUTPUT          : int = auto()
    OUTPUT_NORM     : int = auto()
    ROPE_FREQS      : int = auto()
    ATTN_Q          : int = auto()
    ATTN_K          : int = auto()
    ATTN_V          : int = auto()
    ATTN_QKV        : int = auto()
    ATTN_OUT        : int = auto()
    ATTN_NORM       : int = auto()
    ATTN_NORM_2     : int = auto()
    ATTN_ROT_EMBD   : int = auto()
    FFN_GATE        : int = auto()
    FFN_DOWN        : int = auto()
    FFN_UP          : int = auto()
    FFN_NORM        : int = auto()
    ATTN_Q_NORM     : int = auto()
    ATTN_K_NORM     : int = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:          "llama",
    MODEL_ARCH.FALCON:         "falcon",
    MODEL_ARCH.BAICHUAN:       "baichuan",
    MODEL_ARCH.GPT2:           "gpt2",
    MODEL_ARCH.GPTJ:           "gptj",
    MODEL_ARCH.GPTNEOX:        "gptneox",
    MODEL_ARCH.MPT:            "mpt",
    MODEL_ARCH.STARCODER:      "starcoder",
    MODEL_ARCH.PERSIMMON:      "persimmon",
    MODEL_ARCH.REFACT:         "refact",
    MODEL_ARCH.BERT:           "bert",
    MODEL_ARCH.BLOOM:          "bloom",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:      "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM: "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:     "token_types",
    MODEL_TENSOR.POS_EMBD:        "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:     "output_norm",
    MODEL_TENSOR.OUTPUT:          "output",
    MODEL_TENSOR.ROPE_FREQS:      "rope_freqs",
    MODEL_TENSOR.ATTN_NORM:       "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:     "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:        "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:          "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:          "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:          "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:        "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:   "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_Q_NORM:     "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:     "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.FFN_NORM:        "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_GATE:        "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:        "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:          "blk.{bid}.ffn_up",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT2: [
        # TODO
    ],
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.ROPE_FREQS,
    ]
}

#
# types
#

class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6

class RopeScalingType(Enum):
    NONE   = 'none'
    LINEAR = 'linear'
    YARN   = 'yarn'

class GGMLQuantizationType(IntEnum):
    F32  = 0
    F16  = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, str) or isinstance(val, bytes) or isinstance(val, bytearray):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            print("Unknown type: "+str(type(val)))
            sys.exit()

# Note: Does not support GGML_QKK_64
QK_K = 256
# Items here are (block size, type size)
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32  : (1, 4),
    GGMLQuantizationType.F16  : (1, 2),
    GGMLQuantizationType.Q4_0 : (32, 2 + 16),
    GGMLQuantizationType.Q4_1 : (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0 : (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1 : (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0 : (32, 2 + 32),
    GGMLQuantizationType.Q8_1 : (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K : (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K : (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K : (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K : (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K : (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K : (256, 4 + QK_K + QK_K // 8),
}


# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = KEY.GENERAL.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = KEY.GENERAL.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = KEY.GENERAL.ALIGNMENT
KEY_GENERAL_NAME                 = KEY.GENERAL.NAME
KEY_GENERAL_AUTHOR               = KEY.GENERAL.AUTHOR
KEY_GENERAL_URL                  = KEY.GENERAL.URL
KEY_GENERAL_DESCRIPTION          = KEY.GENERAL.DESCRIPTION
KEY_GENERAL_LICENSE              = KEY.GENERAL.LICENSE
KEY_GENERAL_SOURCE_URL           = KEY.GENERAL.SOURCE_URL
KEY_GENERAL_SOURCE_HF_REPO       = KEY.GENERAL.SOURCE_HF_REPO
KEY_GENERAL_FILE_TYPE            = KEY.GENERAL.FILE_TYPE

# LLM
KEY_CONTEXT_LENGTH        = KEY.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH      = KEY.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT           = KEY.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH   = KEY.LLM.FEED_FORWARD_LENGTH
KEY_USE_PARALLEL_RESIDUAL = KEY.LLM.USE_PARALLEL_RESIDUAL
KEY_TENSOR_DATA_LAYOUT    = KEY.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT        = KEY.ATTENTION.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV     = KEY.ATTENTION.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS    = KEY.ATTENTION.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV         = KEY.ATTENTION.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS     = KEY.ATTENTION.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS = KEY.ATTENTION.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT         = KEY.ROPE.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE               = KEY.ROPE.FREQ_BASE
KEY_ROPE_SCALING_TYPE            = KEY.ROPE.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR          = KEY.ROPE.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN    = KEY.ROPE.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED       = KEY.ROPE.SCALING_FINETUNED

# tokenization
KEY_TOKENIZER_MODEL      = KEY.TOKENIZER.MODEL
KEY_TOKENIZER_LIST       = KEY.TOKENIZER.LIST
KEY_TOKENIZER_TOKEN_TYPE = KEY.TOKENIZER.TOKEN_TYPE
KEY_TOKENIZER_SCORES     = KEY.TOKENIZER.SCORES
KEY_TOKENIZER_MERGES     = KEY.TOKENIZER.MERGES
KEY_TOKENIZER_BOS_ID     = KEY.TOKENIZER.BOS_ID
KEY_TOKENIZER_EOS_ID     = KEY.TOKENIZER.EOS_ID
KEY_TOKENIZER_UNK_ID     = KEY.TOKENIZER.UNK_ID
KEY_TOKENIZER_SEP_ID     = KEY.TOKENIZER.SEP_ID
KEY_TOKENIZER_PAD_ID     = KEY.TOKENIZER.PAD_ID
KEY_TOKENIZER_HF_JSON    = KEY.TOKENIZER.HF_JSON
KEY_TOKENIZER_RWKV       = KEY.TOKENIZER.RWKV
