// Minimal text interaction frontend for PQ-enabled llama.cpp models.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class interaction_mode {
    auto_detect,
    chat,
    completion,
};

struct options {
    std::string model;
    std::string system;
    std::string single_prompt;
    interaction_mode mode = interaction_mode::auto_detect;
    int threads = 60;
    int context = 4096;
    int max_tokens = 256;
    float temperature = 0.7f;
    bool pq_decode = true;
};

struct message {
    std::string role;
    std::string content;
};

void usage(const char * program) {
    std::fprintf(stderr,
        "Usage: %s -m MODEL [options]\n\n"
        "Options:\n"
        "  -t N                       CPU threads (default: 60)\n"
        "  -c N                       context size (default: 4096)\n"
        "  -n N                       maximum response tokens (default: 256)\n"
        "  --temp N                    sampling temperature; 0 is greedy (default: 0.7)\n"
        "  --mode auto|chat|completion\n"
        "                             interaction mode (default: auto)\n"
        "  --system TEXT              system message or completion prefix\n"
        "  --single PROMPT            run one turn and exit\n"
        "  --no-pq                    disable PQ decode for diagnostics\n",
        program);
}

interaction_mode parse_mode(const std::string & value) {
    if (value == "auto") return interaction_mode::auto_detect;
    if (value == "chat") return interaction_mode::chat;
    if (value == "completion") return interaction_mode::completion;
    throw std::runtime_error("--mode must be auto, chat, or completion");
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") {
            result.model = require_value(arg.c_str());
        } else if (arg == "-t" || arg == "--threads") {
            result.threads = std::stoi(require_value(arg.c_str()));
        } else if (arg == "-c" || arg == "--ctx-size") {
            result.context = std::stoi(require_value(arg.c_str()));
        } else if (arg == "-n" || arg == "--max-tokens") {
            result.max_tokens = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--temp") {
            result.temperature = std::stof(require_value("--temp"));
        } else if (arg == "--mode") {
            result.mode = parse_mode(require_value("--mode"));
        } else if (arg == "--system") {
            result.system = require_value("--system");
        } else if (arg == "--single") {
            result.single_prompt = require_value("--single");
        } else if (arg == "--no-pq") {
            result.pq_decode = false;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (result.model.empty()) throw std::runtime_error("a model path is required");
    if (result.threads <= 0) throw std::runtime_error("threads must be positive");
    if (result.context <= 0) throw std::runtime_error("context must be positive");
    if (result.max_tokens <= 0) throw std::runtime_error("max tokens must be positive");
    if (result.temperature < 0.0f) throw std::runtime_error("temperature must be non-negative");
    return result;
}

void log_callback(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR || std::strstr(text, "PQ decode enabled") ||
            std::strstr(text, "PQ skipped tensor")) {
        std::fputs(text, stderr);
    }
}

std::string apply_chat_template(const char * chat_template,
                                const std::vector<message> & messages,
                                bool add_assistant) {
    std::vector<llama_chat_message> raw;
    raw.reserve(messages.size());
    for (const auto & item : messages) {
        raw.push_back({item.role.c_str(), item.content.c_str()});
    }
    int32_t size = llama_chat_apply_template(
        chat_template, raw.data(), raw.size(), add_assistant, nullptr, 0);
    if (size < 0) throw std::runtime_error("the embedded chat template is unsupported");
    std::vector<char> buffer(static_cast<size_t>(size) + 1);
    const int32_t written = llama_chat_apply_template(
        chat_template, raw.data(), raw.size(), add_assistant,
        buffer.data(), static_cast<int32_t>(buffer.size()));
    if (written < 0 || written > static_cast<int32_t>(buffer.size())) {
        throw std::runtime_error("failed to apply the embedded chat template");
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & prompt) {
    int32_t count = llama_tokenize(
        vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
        nullptr, 0, true, true);
    if (count >= 0) throw std::runtime_error("failed to count prompt tokens");
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(
        vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count < 0) throw std::runtime_error("failed to tokenize prompt");
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        const options opt = parse_options(argc, argv);
        llama_log_set(log_callback, nullptr);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.pq_decode = opt.pq_decode;
        llama_model * model = llama_model_load_from_file(opt.model.c_str(), model_params);
        if (!model) throw std::runtime_error("failed to load model: " + opt.model);

        const char * embedded_template = llama_model_chat_template(model, nullptr);
        const bool has_template = embedded_template && embedded_template[0] != '\0';
        bool chat_mode = false;
        switch (opt.mode) {
            case interaction_mode::auto_detect: chat_mode = has_template; break;
            case interaction_mode::chat:
                if (!has_template) {
                    llama_model_free(model);
                    throw std::runtime_error(
                        "chat mode requested, but the GGUF has no tokenizer.chat_template");
                }
                chat_mode = true;
                break;
            case interaction_mode::completion: chat_mode = false; break;
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(opt.context);
        context_params.n_batch = static_cast<uint32_t>(std::min(opt.context, 512));
        context_params.n_threads = opt.threads;
        context_params.n_threads_batch = opt.threads;
        context_params.offload_kqv = false;
        context_params.no_perf = false;
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) {
            llama_model_free(model);
            throw std::runtime_error("failed to create llama context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_sampler * sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
        if (opt.temperature == 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(opt.temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        std::vector<message> history;
        auto clear = [&]() {
            history.clear();
            if (chat_mode && !opt.system.empty()) {
                history.push_back({"system", opt.system});
            }
            llama_memory_clear(llama_get_memory(context), true);
            llama_sampler_reset(sampler);
        };
        clear();

        std::fprintf(stderr, "[MODEL] %s\n", opt.model.c_str());
        std::fprintf(stderr, "[PQ] %s\n", opt.pq_decode ? "enabled" : "disabled");
        std::fprintf(stderr, "[TEMP] %.3g%s\n", opt.temperature,
                     opt.temperature == 0.0f ? " (greedy)" : "");
        std::fprintf(stderr, "[MODE] %s%s\n",
                     chat_mode ? "chat: embedded tokenizer.chat_template" :
                                 "completion: no template applied",
                     opt.mode == interaction_mode::auto_detect ? " (auto)" : " (forced)");
        if (!chat_mode && !opt.system.empty()) {
            std::fprintf(stderr,
                "[MODE] completion treats --system as a plain-text prefix\n");
        }

        auto run_turn = [&](const std::string & user) {
            std::string prompt;
            if (chat_mode) {
                history.push_back({"user", user});
                prompt = apply_chat_template(embedded_template, history, true);
            } else {
                prompt = opt.system.empty() ? user : opt.system + "\n\n" + user;
            }

            // Re-render and decode the complete prompt on every turn. This is
            // slower than relying on template prefix stability, but correct
            // for templates whose output depends on the full message list.
            llama_memory_clear(llama_get_memory(context), true);
            llama_sampler_reset(sampler);
            std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
            if (prompt_tokens.size() + static_cast<size_t>(opt.max_tokens) >
                    static_cast<size_t>(llama_n_ctx(context))) {
                throw std::runtime_error("prompt and response exceed the context size");
            }

            llama_perf_context_reset(context);
            llama_batch batch = llama_batch_get_one(
                prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
            std::string response;
            int generated = 0;
            while (generated < opt.max_tokens) {
                if (llama_decode(context, batch) != 0) {
                    throw std::runtime_error("llama_decode failed");
                }
                llama_token id = llama_sampler_sample(sampler, context, -1);
                if (llama_vocab_is_eog(vocab, id)) break;
                char piece[256];
                const int32_t size = llama_token_to_piece(
                    vocab, id, piece, sizeof(piece), 0, false);
                if (size < 0) throw std::runtime_error("failed to decode token");
                response.append(piece, static_cast<size_t>(size));
                std::fwrite(piece, 1, static_cast<size_t>(size), stdout);
                std::fflush(stdout);
                ++generated;
                batch = llama_batch_get_one(&id, 1);
            }
            std::fputc('\n', stdout);
            if (chat_mode) history.push_back({"assistant", response});

            const llama_perf_context_data perf = llama_perf_context(context);
            const double prompt_tps = perf.t_p_eval_ms > 0
                ? 1000.0 * perf.n_p_eval / perf.t_p_eval_ms : 0.0;
            const double generation_tps = perf.t_eval_ms > 0
                ? 1000.0 * perf.n_eval / perf.t_eval_ms : 0.0;
            std::printf("[Prompt: %.1f t/s | Generation: %.1f t/s | Tokens: %d]\n",
                        prompt_tps, generation_tps, generated);
        };

        if (!opt.single_prompt.empty()) {
            std::printf(chat_mode ? "User: %s\nAssistant: " :
                                    "Prompt: %s\nCompletion: ",
                        opt.single_prompt.c_str());
            run_turn(opt.single_prompt);
        } else {
            std::puts(chat_mode
                ? "EdgePQ chat ready. Commands: /clear, /exit"
                : "EdgePQ completion ready. Each prompt is independent. Commands: /clear, /exit");
            while (true) {
                std::printf(chat_mode ? "\nUser> " : "\nPrompt> ");
                std::fflush(stdout);
                std::string user;
                if (!std::getline(std::cin, user) || user == "/exit" || user == "/quit") {
                    break;
                }
                if (user == "/clear") {
                    clear();
                    std::puts(chat_mode ? "Conversation cleared." : "Completion state cleared.");
                    continue;
                }
                if (user.empty()) continue;
                std::printf(chat_mode ? "Assistant> " : "Completion> ");
                std::fflush(stdout);
                run_turn(user);
            }
        }

        llama_sampler_free(sampler);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
