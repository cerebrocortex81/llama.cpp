#include <string>
#include <vector>
#include <sstream>

#undef NDEBUG
#include <cassert>

#include "common.h"

int main(void) {
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    gpt_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            gpt_params_parser_init(params, (enum llama_example)ex);
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;
    auto options = gpt_params_parser_init(params, LLAMA_EXAMPLE_COMMON);

    printf("test-arg-parser: test invalid usage\n\n");

    argv = {"binary_name", "-m"};
    assert(false == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));

    argv = {"binary_name", "-ngl", "hello"};
    assert(false == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));

    argv = {"binary_name", "-sm", "hello"};
    assert(false == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.model == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.verbosity == 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.model == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.model == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);


    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == gpt_params_parse(argv.size(), list_str_to_char(argv).data(), params, options));
    assert(params.model == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);


    printf("test-arg-parser: all tests OK\n\n");
#endif // __MINGW32__
}
