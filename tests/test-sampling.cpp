#include "ggml.h"
#include "llama.h"
#include <assert.h>
#include <math.h>
#include <numeric>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

void dump(const llama_token_data_array * candidates) {
    for (size_t i = 0; i < candidates->size; i++) {
        printf("%d: %f (%f)\n", candidates->data[i].id, candidates->data[i].p, candidates->data[i].logit);
    }
}

#define DUMP(__candidates) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__candidates)); printf("-\n"); } while(0)


void test_top_k(const std::vector<float> & probs,
                const std::vector<float> & expected_probs,
                int k) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    llama_sample_softmax(nullptr, &candidates_p);
    // DUMP(&candidates_p);
    llama_sample_top_k(nullptr, &candidates_p, k);
    // DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-5);
    }
}


void test_top_p(const std::vector<float> & probs,
                const std::vector<float> & expected_probs,
                float p) {

    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    // DUMP(&candidates_p);
    llama_sample_top_p(nullptr, &candidates_p, p);
    // DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-5);
    }
}


void test_tfs(const std::vector<float> & probs,
                const std::vector<float> & expected_probs,
                float z) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    // DUMP(&candidates_p);
    llama_sample_tail_free(nullptr, &candidates_p, z);
    // DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-6);
    }
}


void test_typical(const std::vector<float> & probs,
                const std::vector<float> & expected_probs,
                float p) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    // DUMP(&candidates_p);
    llama_sample_typical(nullptr, &candidates_p, p);
    // DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-6);
    }
}


void test_repetition_penalty(
                const std::vector<float> & probs,
                const std::vector<llama_token> & last_tokens,
                const std::vector<float> & expected_probs,
                float penalty) {
    assert(probs.size() == expected_probs.size());

    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_repetition_penalty(nullptr, &candidates_p, (llama_token *)last_tokens.data(), last_tokens.size(), penalty);
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-6);
    }
}


void test_frequency_presence_penalty(
                const std::vector<float> & probs,
                const std::vector<llama_token> & last_tokens,
                const std::vector<float> & expected_probs,
                float alpha_frequency, float alpha_presence) {
    assert(probs.size() == expected_probs.size());

    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    llama_sample_softmax(nullptr, &candidates_p);
    // DUMP(&candidates_p);
    llama_sample_frequency_and_presence_penalties(nullptr, &candidates_p, (llama_token *)last_tokens.data(), last_tokens.size(), alpha_frequency, alpha_presence);
    llama_sample_softmax(nullptr, &candidates_p);
    // DUMP(&candidates_p);

    assert(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-6);
    }
}


void test_mirostat() {
    std::vector<float> probs = {0.1, 0.2, 0.3, 0.4};
    std::vector<float> expected_probs = {0.1, 0.2, 0.3, 0.4};

    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit = log(probs[i]);
        candidates.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size() };
    DUMP(&candidates_p);

    float tau = 5.0f;
    float mu = 2.0f * tau;
    int k = 0;
    float eta = 0.1f;
    int m = 100;
    // float N = 32000;
    float N = 4;
    // llama_sample_mirostat(ctx, &candidates_p, tau, eta, m, N, &k, &mu);
    DUMP(&candidates_p);

    // assert(candidates_p.size == expected_probs.size());
    // for (size_t i = 0; i < candidates_p.size; i++) {
    //     assert(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-6);
    // }
}

int main(void) {
    test_top_k({0.1, 0.2, 0.3, 0.4}, {0.4}, 1);
    test_top_k({0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2}, 3);

    test_top_p({0.1, 0.2, 0.3, 0.4}, {0.4}, 0);
    test_top_p({0.1, 0.2, 0.3, 0.4}, {0.4, 0.3}, 0.7);
    test_top_p({0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}, 1);

    test_tfs({0.1, 0.15, 0.2, 0.25, 0.3}, {0.3}, 0.25);
    test_tfs({0.1, 0.15, 0.2, 0.25, 0.3}, {0.3, 0.25}, 0.75);
    test_tfs({0.1, 0.15, 0.2, 0.25, 0.3}, {0.3, 0.25}, 0.99);

    test_typical({0.97, 0.01, 0.01, 0.01}, {0.97}, 0.5);
    test_typical({0.4, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}, 0.5);

    test_repetition_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0}, {0, 0.25, 0.25, 0.25, 0.25}, 50.0);
    test_repetition_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0, 1, 2}, {0, 0, 0, 0.5, 0.5}, 50.0);
    test_repetition_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5, 0.5}, 50.0);

    test_frequency_presence_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0},             {0.249997, 0.249997, 0.249997, 0.249997, 0.000011}, 5.0, 5.0);
    test_frequency_presence_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0, 1, 2},       {0.499966, 0.499966, 0.000023, 0.000023, 0.000023}, 5.0, 5.0);
    test_frequency_presence_penalty({0.2, 0.2, 0.2, 0.2, 0.2}, {0, 1, 2, 0, 0}, {0.499977, 0.499977, 0.000023, 0.000023, 0.000000}, 5.0, 5.0);

    // test_mirostat();

    printf("OK\n");
}
