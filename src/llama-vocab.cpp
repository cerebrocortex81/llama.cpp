#include "llama-vocab.h"

#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <queue>
#include <sstream>

//
// helpers
//

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

struct naive_trie {
    naive_trie() : has_value(false), value(0) {
    }
    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        } else {
            return std::make_pair(key, offset);
        }
    }
    struct naive_trie * traverse(const char c) {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        } else {
            return NULL;
        }
    }
    std::map<char, struct naive_trie> children;
    bool has_value;
    llama_token value;
};

//
// impl
//

int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ')   == std::string::npos);
    GGML_ASSERT(token_left.find('\n')  == std::string::npos);
    GGML_ASSERT(token_right.find(' ')  == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_CONTROL;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_BYTE;
}

static bool llama_is_user_defined_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_USER_DEFINED;
}

static bool llama_is_unused_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNUSED;
}

static uint8_t llama_token_to_byte(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto & token_data = vocab.id_to_token.at(id);
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            GGML_ABORT("fatal error");
            //return unicode_utf8_to_byte(token_data.text); // TODO: why is this here after GGML_ASSERT?
        }
        case LLAMA_VOCAB_TYPE_WPM: {
            GGML_ABORT("fatal error");
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
//

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token_impl(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe {
    llm_tokenizer_bpe(const llama_vocab & vocab): vocab(vocab) {
        GGML_ASSERT(vocab.type == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.type_pre) {
            case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DBRX:
            case LLAMA_VOCAB_PRE_TYPE_SMAUG:
                regex_exprs = {
                    // same as llama3
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                    "\\s+$",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?\\p{L}+",
                    "\\s?\\p{P}+",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_FALCON:
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STARCODER:
            case LLAMA_VOCAB_PRE_TYPE_REFACT:
            case LLAMA_VOCAB_PRE_TYPE_COMMAND_R:
            case LLAMA_VOCAB_PRE_TYPE_SMOLLM:
            case LLAMA_VOCAB_PRE_TYPE_CODESHELL:
                regex_exprs = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_GPT2:
            case LLAMA_VOCAB_PRE_TYPE_MPT:
            case LLAMA_VOCAB_PRE_TYPE_OLMO:
            case LLAMA_VOCAB_PRE_TYPE_JAIS:
                regex_exprs = {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
            case LLAMA_VOCAB_PRE_TYPE_QWEN2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_PORO:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
                regex_exprs = {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_VIKING:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_TEKKEN:
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    void append(const llama_vocab::id token_id, std::vector<llama_vocab::id> & output) const {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_bos) {
            GGML_ASSERT(vocab.special_bos_id != -1);
            output.push_back(vocab.special_bos_id);
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_eos) {
            GGML_ASSERT(vocab.special_eos_id != -1);
            output.push_back(vocab.special_eos_id);
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.tokenizer_add_eos && output.size() >= 2 && *(output.end()-2) == vocab.special_eos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        int final_prev_index = -1;

        const auto word_collection = unicode_regex_split(text, regex_exprs);

        symbols_final.clear();

        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte != vocab.token_to_id.end()) {
                            output.push_back(token_multibyte->second);
                        }
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const llama_vocab & vocab;

    std::vector<std::string> regex_exprs;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

//
// WPM tokenizer
//

struct llm_tokenizer_wpm {
    llm_tokenizer_wpm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) const {
        const auto & token_map = vocab.token_to_id;

        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);

        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len + 1); j > i; j--) {
                    auto it = token_map.find(word1.substr(i, j - i));
                    if (it != token_map.end()) {
                        output.push_back(it->second);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.special_unk_id);
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    std::vector<std::string> preprocess(const std::string & text) const {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto categ = unicode_cpt_category(cpt);

            if (categ.is_whitespace()) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            if (cpt == 0 || cpt == 0xFFFD || categ.is_C()) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (categ.is_P() || (cpt < 0x7F && categ.is_S()) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {  //TODO: move to unicode-data.cpp? unicode_cpt_category(cpt).is_chinese()?
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
            //(cpt >= 0x3000  && cpt <= 0x303F)  ||
            //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

    const llama_vocab & vocab;
};

//
// UGM tokenizer
//

struct llm_tokenizer_ugm {
    llm_tokenizer_ugm(const llama_vocab & vocab) : vocab(vocab) {
        if (vocab.precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &vocab.precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= vocab.precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t *) &vocab.precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &vocab.precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = vocab.precompiled_charsmap.size() - charsmap_offset;
        }

        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto &token_data = vocab.id_to_token[id];

            if (llama_is_normal_token(vocab, id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (llama_is_normal_token(vocab, id) ||
                llama_is_user_defined_token(vocab, id) ||
                llama_is_unused_token(vocab, id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (llama_is_user_defined_token(vocab, id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.special_unk_id, 0, -FLT_MAX});
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.special_unk_id, 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];
            struct naive_trie * node  = token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llama_token token_id = node->value;
                    const auto & token_data = vocab.id_to_token[token_id];

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.special_unk_id, input_offset, (float) challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.special_unk_id;
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin(), output.end());
    }

private:
    const llama_vocab & vocab;

    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.tokenizer_escape_whitespaces ? escaped_space : " ";

        bool shall_prepend_space = !vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_append_space = vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_merge_spaces = vocab.tokenizer_remove_extra_whitespaces;

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Eeach bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t * xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index > xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t * xcda_array;
        size_t xcda_array_size;
    };

    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match = user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (xcda_array_size > 0) {
            struct xcda_array_view xcda_view(xcda_array, xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &prefix_replacements[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        } else {
            // check if the input prefix contains a valid sequence of UTF-8 code units
            try {
                // if yes, return this sequence unmodified
                size_t prefix_offset = input_offset;
                unicode_cpt_from_utf8(input, prefix_offset);
                return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
            } catch (std::invalid_argument & /*ex*/) {
                // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
                return { "\xEF\xBF\xBD", 3, 1 };
            }
        }
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char * prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t * xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        llama_token token_id;
        size_t input_offset;
        float score_sum;
    };

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

//
// (de-) tokenize
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_vocab::id _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

// #define PRETOKENIZERDEBUG

static void tokenizer_st_partition(const llama_vocab & vocab, std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) {
    // for each special token
    for (const llama_vocab::id special_id : vocab.cache_special_tokens) {
        const auto & data = vocab.id_to_token[special_id];
        const auto & special_token = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text.find(special_token, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + special_token.length() > raw_text_base_offset + raw_text_base_length) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + special_token.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + special_token.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, std::string raw_text, bool add_special, bool parse_special) {
    std::vector<llama_vocab::id> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(vocab, fragment_buffer, parse_special);
    }

    switch (vocab.type) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []

                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && vocab.tokenizer_add_bos) {
                    GGML_ASSERT(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                    is_prev_special = true;
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

                        // prefix with space if previous is special
                        if (vocab.tokenizer_add_space_prefix && is_prev_special) {
                            raw_text = " " + raw_text;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_spm tokenizer(vocab);
                        llama_escape_whitespace(raw_text);
                        tokenizer.tokenize(raw_text, output);
                        is_prev_special = false;
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
                }

                if (add_special && vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
                    LLAMA_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && vocab.tokenizer_add_eos) {
                    GGML_ASSERT(vocab.special_eos_id != -1);
                    output.push_back(vocab.special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_BPE:
            {
                llm_tokenizer_bpe tokenizer(vocab);

                if (add_special) {
                    tokenizer.append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        tokenizer.append(fragment.token, output);
                    }
                }

                if (add_special) {
                    tokenizer.append_eos(output);
                    tokenizer.check_double_bos_eos(output);
                }
            } break;
        case LLAMA_VOCAB_TYPE_WPM:
            {
                if (add_special) {
                    GGML_ASSERT(vocab.special_cls_id != -1);
                    output.push_back(vocab.special_cls_id);
                }

                llm_tokenizer_wpm tokenizer(vocab);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special) {
                    GGML_ASSERT(vocab.special_sep_id != -1);
                    output.push_back(vocab.special_sep_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_UGM:
            {
                llm_tokenizer_ugm tokenizer(vocab);

                if (add_special && vocab.tokenizer_add_bos != 0) {
                    GGML_ASSERT(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && vocab.tokenizer_add_bos != 0 && output.size() >= 2 && output[1] == vocab.special_bos_id) {
                    LLAMA_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && vocab.tokenizer_add_eos == 1) {
                    GGML_ASSERT(vocab.special_eos_id != -1);
                    output.push_back(vocab.special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_NONE:
            GGML_ABORT("fatal error");
    }

    return output;
}

llama_token llama_byte_to_token_impl(const llama_vocab & vocab, uint8_t ch) {
    GGML_ASSERT(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = vocab.token_to_id.find(buf);
            if (token != vocab.token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return vocab.token_to_id.at(buf2);
        }
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_BPE: {
            return vocab.token_to_id.at(unicode_byte_to_utf8(ch));
        }
        default:
            GGML_ABORT("fatal error");
    }
}

const char * llama_token_get_text_impl(const struct llama_vocab & vocab, llama_token token) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].text.c_str();
}

float llama_token_get_score_impl(const struct llama_vocab & vocab, llama_token token) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].score;
}

llama_token_attr llama_token_get_attr_impl(const struct llama_vocab & vocab, llama_token token) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].attr;
}

bool llama_token_is_eog_impl(const struct llama_vocab & vocab, llama_token token) {
    return token != -1 && (
        token == llama_token_eos_impl(vocab) ||
        token == llama_token_eot_impl(vocab)
    );
}

bool llama_token_is_control_impl(const struct llama_vocab & vocab, llama_token token) {
    return llama_is_control_token(vocab, token);
}

llama_token llama_token_bos_impl(const struct llama_vocab & vocab) {
    return vocab.special_bos_id;
}

llama_token llama_token_eos_impl(const struct llama_vocab & vocab) {
    return vocab.special_eos_id;
}

llama_token llama_token_cls_impl(const struct llama_vocab & vocab) {
    return vocab.special_cls_id;
}

llama_token llama_token_sep_impl(const struct llama_vocab & vocab) {
    return vocab.special_sep_id;
}

llama_token llama_token_nl_impl(const struct llama_vocab & vocab) {
    return vocab.linefeed_id;
}

llama_token llama_token_pad_impl(const struct llama_vocab & vocab) {
    return vocab.special_pad_id;
}

int32_t llama_add_bos_token_impl(const struct llama_vocab & vocab) {
    return vocab.tokenizer_add_bos;
}

int32_t llama_add_eos_token_impl(const struct llama_vocab & vocab) {
    return vocab.tokenizer_add_eos;
}

llama_token llama_token_prefix_impl(const struct llama_vocab & vocab) {
    return vocab.special_prefix_id;
}

llama_token llama_token_middle_impl(const struct llama_vocab & vocab) {
    return vocab.special_middle_id;
}

llama_token llama_token_suffix_impl(const struct llama_vocab & vocab) {
    return vocab.special_suffix_id;
}

llama_token llama_token_eot_impl(const struct llama_vocab & vocab) {
    return vocab.special_eot_id;
}

int32_t llama_tokenize_impl(
    const struct llama_vocab & vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    auto res = llama_tokenize_internal(vocab, std::string(text, text_len), add_special, parse_special);
    if (n_tokens_max < (int) res.size()) {
        // LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

// does not write null-terminator to buf
int32_t llama_token_to_piece_impl(const struct llama_vocab & vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) {
    // ref: https://github.com/ggerganov/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
    const llama_token_attr attr = llama_token_get_attr_impl(vocab, token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = vocab.cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) vocab.id_to_token.size()) {
        const std::string & token_text = vocab.id_to_token[token].text;
        switch (llama_vocab_get_type(vocab)) {
            case LLAMA_VOCAB_TYPE_WPM:
            case LLAMA_VOCAB_TYPE_SPM:
            case LLAMA_VOCAB_TYPE_UGM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                } else if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = token_text;
                    llama_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                } else if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                    char byte = (char) llama_token_to_byte(vocab, token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                } else if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = llama_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                break;
            }
            default:
                GGML_ABORT("fatal error");
        }
    }

    return 0;
}

int32_t llama_detokenize_impl(
        const struct llama_vocab & vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) {
    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = vocab.tokenizer_add_space_prefix;

    if (remove_special && vocab.tokenizer_add_bos) {
        if (n_tokens > 0 && tokens[0] == vocab.special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && vocab.tokenizer_add_eos) {
        if (n_tokens > 0 && tokens[n_tokens-1] == vocab.special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(avail >= 0);
        int32_t n_chars = llama_token_to_piece_impl(vocab, tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (vocab.tokenizer_clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}
