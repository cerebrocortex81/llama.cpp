#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <climits>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

struct Argument {
    std::string flag;
    std::string help_text;
};

class ArgumentParser {
   public:
    ArgumentParser(const char * program_name) : program_name(program_name) {}

    void add_argument(const std::string & flag, std::string & var, const std::string & help_text = "") {
        string_args[flag] = &var;
        arguments.push_back({flag, help_text});
    }

    void add_argument(const std::string & flag, int & var, const std::string & help_text = "") {
        int_args[flag] = &var;
        arguments.push_back({flag, help_text});
    }

    int parse(int argc, const char ** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (string_args.count(arg)) {
                if (i + 1 < argc) {
                    *string_args[arg] = argv[++i];
                } else {
                    fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                    print_usage();
                    return 1;
                }
            } else if (int_args.count(arg)) {
                if (i + 1 < argc) {
                    if (parse_int_arg(argv[++i], *int_args[arg]) != 0) {
                        fprintf(stderr, "error: invalid value for %s: %s\n", arg.c_str(), argv[i]);
                        print_usage();
                        return 1;
                    }
                } else {
                    fprintf(stderr, "error: missing value for %s\n", arg.c_str());
                    print_usage();
                    return 1;
                }
            } else {
                fprintf(stderr, "error: unrecognized argument %s\n", arg.c_str());
                print_usage();
                return 1;
            }
        }

        if (string_args["-m"]->empty()) {
            fprintf(stderr, "error: -m is required\n");
            print_usage();
            return 1;
        }

        return 0;
    }

   private:
    const char * program_name;
    std::unordered_map<std::string, std::string *> string_args;
    std::unordered_map<std::string, int *> int_args;
    std::vector<Argument> arguments;

    int parse_int_arg(const char * arg, int & value) {
        char * end;
        const long val = std::strtol(arg, &end, 10);
        if (*end == '\0' && val >= INT_MIN && val <= INT_MAX) {
            value = static_cast<int>(val);
            return 0;
        }
        return 1;
    }

    void print_usage() const {
        printf("\nUsage:\n");
        printf("  %s [OPTIONS]\n\n", program_name);
        printf("Options:\n");
        for (const auto & arg : arguments) {
            printf("  %-10s %s\n", arg.flag.c_str(), arg.help_text.c_str());
        }
        printf("\n");
    }
};

// Add a message to `messages` and store its content in `owned_content`
static void add_message(const char * role, const std::string & text, std::vector<llama_chat_message> & messages,
                        std::vector<std::unique_ptr<char[]>> & owned_content) {
    auto content = std::unique_ptr<char[]>(new char[text.size() + 1]);
    std::strcpy(content.get(), text.c_str());
    messages.push_back({role, content.get()});
    owned_content.push_back(std::move(content));
}

// Function to apply the chat template and resize `formatted` if needed
static int apply_chat_template(const llama_model * model, const std::vector<llama_chat_message> & messages,
                               std::vector<char> & formatted, const bool append) {
    int result = llama_chat_apply_template(model, nullptr, messages.data(), messages.size(), append, formatted.data(),
                                           formatted.size());
    if (result > static_cast<int>(formatted.size())) {
        formatted.resize(result);
        result = llama_chat_apply_template(model, nullptr, messages.data(), messages.size(), append, formatted.data(),
                                           formatted.size());
    }

    return result;
}

// Function to tokenize the prompt
static int tokenize_prompt(const llama_model * model, const std::string & prompt,
                           std::vector<llama_token> & prompt_tokens) {
    const int n_prompt_tokens = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    prompt_tokens.resize(n_prompt_tokens);
    if (llama_tokenize(model, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) <
        0) {
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    return n_prompt_tokens;
}

// Check if we have enough space in the context to evaluate this batch
static int check_context_size(const llama_context * ctx, const llama_batch & batch) {
    const int n_ctx = llama_n_ctx(ctx);
    const int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        printf("\033[0m\n");
        fprintf(stderr, "context size exceeded\n");
        return 1;
    }

    return 0;
}

// convert the token to a string
static int convert_token_to_string(const llama_model * model, const llama_token token_id, std::string & piece) {
    char buf[256];
    int n = llama_token_to_piece(model, token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        GGML_ABORT("failed to convert token to piece\n");
    }

    piece = std::string(buf, n);
    return 0;
}

static void print_word_and_concatenate_to_response(const std::string & piece, std::string & response) {
    printf("%s", piece.c_str());
    fflush(stdout);
    response += piece;
}

// helper function to evaluate a prompt and generate a response
static int generate(const llama_model * model, llama_sampler * smpl, llama_context * ctx, const std::string & prompt,
                    std::string & response) {
    std::vector<llama_token> prompt_tokens;
    const int n_prompt_tokens = tokenize_prompt(model, prompt, prompt_tokens);
    if (n_prompt_tokens < 0) {
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    while (true) {
        check_context_size(ctx, batch);
        if (llama_decode(ctx, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token, check is it an end of generation?
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_token_is_eog(model, new_token_id)) {
            break;
        }

        std::string piece;
        if (convert_token_to_string(model, new_token_id, piece)) {
            return 1;
        }

        print_word_and_concatenate_to_response(piece, response);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return 0;
}

static int parse_arguments(const int argc, const char ** argv, std::string & model_path, std::string & prompt,
                           int & n_ctx, int & ngl) {
    ArgumentParser parser(argv[0]);
    parser.add_argument("-m", model_path, "model");
    parser.add_argument("-p", prompt, "prompt");
    parser.add_argument("-c", n_ctx, "context_size");
    parser.add_argument("-ngl", ngl, "n_gpu_layers");
    if (parser.parse(argc, argv)) {
        return 1;
    }

    return 0;
}

static int read_user_input(std::string & user) {
    std::getline(std::cin, user);
    return user.empty();  // Indicate an error or empty input
}

// Function to generate a response based on the prompt
static int generate_response(llama_model * model, llama_sampler * sampler, llama_context * context,
                             const std::string & prompt, std::string & response) {
    // Set response color
    printf("\033[33m");
    if (generate(model, sampler, context, prompt, response)) {
        fprintf(stderr, "failed to generate response\n");
        return 1;
    }

    // End response with color reset and newline
    printf("\n\033[0m");
    return 0;
}

// Helper function to apply the chat template and handle errors
static int apply_chat_template_with_error_handling(llama_model * model, std::vector<llama_chat_message> & messages,
                                                   std::vector<char> & formatted, bool is_user_input,
                                                   int & output_length) {
    int new_len = apply_chat_template(model, messages, formatted, is_user_input);
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return -1;
    }

    output_length = new_len;
    return 0;
}

// Helper function to handle user input
static bool handle_user_input(std::string & user_input, const std::string & prompt_non_interactive) {
    if (!prompt_non_interactive.empty()) {
        user_input = prompt_non_interactive;
        return true;  // No need for interactive input
    }

    printf("\033[32m> \033[0m");
    return !read_user_input(user_input);  // Returns false if input ends the loop
}

// The main chat loop where user inputs are processed and responses generated.
static int chat_loop(llama_model * model, llama_sampler * sampler, llama_context * context,
                     std::string & prompt_non_interactive) {
    std::vector<llama_chat_message> messages;
    std::vector<std::unique_ptr<char[]>> owned_content;
    std::vector<char> fmtted(llama_n_ctx(context));
    int prev_len = 0;
    while (true) {
        // Get user input
        std::string user_input;
        if (!handle_user_input(user_input, prompt_non_interactive)) {
            break;
        }

        add_message("user", prompt_non_interactive.empty() ? user_input : prompt_non_interactive, messages,
                    owned_content);
        int new_len;
        if (apply_chat_template_with_error_handling(model, messages, fmtted, true, new_len) < 0) {
            return 1;
        }

        std::string prompt(fmtted.begin() + prev_len, fmtted.begin() + new_len);
        std::string response;
        if (generate_response(model, sampler, context, prompt, response)) {
            return 1;
        }

        if (!prompt_non_interactive.empty()) {
            return 0;
        }

        add_message("assistant", response, messages, owned_content);
        prev_len = apply_chat_template(model, messages, fmtted, false);
        if (apply_chat_template_with_error_handling(model, messages, fmtted, false, prev_len) < 0) {
            return 1;
        }
    }

    return 0;
}

static void log_callback(const enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

// Initializes the model and returns a unique pointer to it.
static std::unique_ptr<llama_model, decltype(&llama_free_model)> initialize_model(const std::string & model_path,
                                                                                  int ngl) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    auto model = std::unique_ptr<llama_model, decltype(&llama_free_model)>(
        llama_load_model_from_file(model_path.c_str(), model_params), llama_free_model);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
    }

    return model;
}

// Initializes the context with the specified parameters.
static std::unique_ptr<llama_context, decltype(&llama_free)> initialize_context(llama_model * model, int n_ctx) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    auto context = std::unique_ptr<llama_context, decltype(&llama_free)>(
        llama_new_context_with_model(model, ctx_params), llama_free);
    if (!context) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
    }

    return context;
}

// Initializes and configures the sampler.
static std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> initialize_sampler() {
    auto sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return sampler;
}

static bool is_stdin_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    return GetConsoleMode(hStdin, &mode);
#else
    return isatty(STDIN_FILENO);
#endif
}

static std::string read_pipe_data() {
    std::ostringstream result;
    result << std::cin.rdbuf();  // Read all data from std::cin
    return result.str();
}

int main(int argc, const char ** argv) {
    std::string model_path, prompt_non_interactive;
    int ngl = 99;
    int n_ctx = 2048;
    if (parse_arguments(argc, argv, model_path, prompt_non_interactive, n_ctx, ngl)) {
        return 1;
    }

    if (!is_stdin_a_terminal()) {
        if (!prompt_non_interactive.empty()) {
            prompt_non_interactive += "\n\n";
        }

        prompt_non_interactive += read_pipe_data();
    }

    llama_log_set(log_callback, nullptr);
    auto model = initialize_model(model_path, ngl);
    if (!model) {
        return 1;
    }

    auto context = initialize_context(model.get(), n_ctx);
    if (!context) {
        return 1;
    }

    auto sampler = initialize_sampler();
    std::vector<llama_chat_message> messages;
    if (chat_loop(model.get(), sampler.get(), context.get(), prompt_non_interactive)) {
        return 1;
    }

    return 0;
}
