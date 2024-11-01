@llama.cpp
@num_tokens
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file ../../../test-model.gguf
    And   a model alias tinyllama-2
    And   42 as server seed
    And   256 KV cache size
    And   2 slots
    And   -2 max tokens to predict

    Then  the server is starting
    Then  the server is healthy
    


  Scenario: Generate tokens until context is full
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Tell me a long story?
    """
    And   a completion request with no api error
    Then  1 tokens are predicted 