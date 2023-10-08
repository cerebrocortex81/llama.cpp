#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <unordered_map>

// sampling parameters
typedef struct llama_sampling_params {
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    bool    penalize_nl       = true;  // consider newlines as a repeatable token

    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt;   // string to help guidance
    float       cfg_scale     = 1.f;   // How strong is guidance

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

} llama_sampling_params;

// per-sequence sampler state
typedef struct llama_sampler_sequence_state {
    float mirostat_mu; // mirostat sampler state
    llama_grammar * grammar;
} llama_sampler_sequence_state;

// general sampler state
typedef struct llama_sampling_state {
    ~llama_sampling_state();

    // parameters that will be used for sampling and when creating
    // new llama_sampler_sequence_state instances
    llama_sampling_params params;

    // map of sequence ids to sampler states
    std::unordered_map<llama_seq_id, llama_sampler_sequence_state> sequence_states;

    // when non-NULL, new instances of llama_sampler_sequence_state
    // will get a copy of the grammar here
    // note: only the pointer is stored here, it is not a copy of
    //       the grammar and shouldn't be freed
    llama_grammar * grammar;
} llama_sampling_state;

#include "common.h"

// Create a new sampling state instance.
llama_sampling_state llama_sampling_state_init(
        const struct gpt_params & params,
                  llama_grammar * grammar = NULL);

// Fetches the sampler state for the specified sequence id (defaults to 0).
// If the state for that sequence id doesn't already exist, it will be created with
// default values based on the parameters in the state argument.
llama_sampler_sequence_state & llama_sampling_get_sequence_state(
              llama_sampling_state & state,
        const llama_seq_id           seq = 0);

// Reset the sampler states for the supplied sequence id (defaults to 0).
// This is necessary to reuse a sequence id or free memory used by sequences
// that are no longer required.
bool llama_sampling_state_reset(
              llama_sampling_state & state,
        const llama_seq_id           seq = 0);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to delete
//       the item in params.sampler_state when a sequence ends and samplers that rely on
//       state are being used.
//
// required:
//  - ctx:    context to use for sampling
//  - params: sampling parameters
//
// optional:
//  - ctx_guidance:  context to use for classifier-free guidance, ignore if NULL
//  - grammar:       grammar to use for sampling, ignore if NULL
//  - last_tokens:   needed for repetition penalty, ignore if empty
//  - idx:           sample from llama_get_logits_ith(ctx, idx)
//  - seq:           sequence id to associate sampler state with
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sample_token(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_sampling_state & state,
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
        const                      int   idx = 0,
                          llama_seq_id   seq = 0);
