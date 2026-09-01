#include "llama.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string fp16_model;
    std::string q2_model;
    std::string pq_model;
    std::string output;
    int threads = 60;
    int repetitions = 3;
    int warmup = 1;
    int context = 2048;
    std::vector<int> generations{128, 512};
    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
};

struct Measurement {
    std::string name;
    std::string path;
    int generation = 0;
    double mean = 0.0;
    double stdev = 0.0;
};

static void usage(const char * program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Required:\n"
        << "  --fp16 PATH              F16 GGUF model\n"
        << "  --q2 PATH                Q2_K GGUF model\n"
        << "  --pq PATH                PQ-4c8b GGUF model\n\n"
        << "Options:\n"
        << "  --threads N              CPU threads (default: 60)\n"
        << "  --repetitions N          measured repetitions (default: 3)\n"
        << "  --warmup N               warmup repetitions (default: 1)\n"
        << "  --generations LIST       comma-separated lengths (default: 128,512)\n"
        << "  --context N              context size (default: 2048)\n"
        << "  --numa MODE              distribute, isolate, or disabled\n"
        << "  --output PATH            CSV output path\n"
        << "  -h, --help               show this help\n";
}

static std::string require_value(int & index, int argc, char ** argv, const char * option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
    }
    return argv[++index];
}

static std::vector<int> parse_int_list(const std::string & text) {
    std::vector<int> values;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string token = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int value = std::stoi(token);
        if (value <= 0) {
            throw std::runtime_error("list values must be positive: " + text);
        }
        values.push_back(value);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("empty integer list");
    }
    return values;
}

static ggml_numa_strategy parse_numa(const std::string & value) {
    if (value == "distribute") {
        return GGML_NUMA_STRATEGY_DISTRIBUTE;
    }
    if (value == "isolate") {
        return GGML_NUMA_STRATEGY_ISOLATE;
    }
    if (value == "disabled") {
        return GGML_NUMA_STRATEGY_DISABLED;
    }
    throw std::runtime_error("unsupported NUMA mode: " + value);
}

static Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--fp16") {
            options.fp16_model = require_value(i, argc, argv, "--fp16");
        } else if (arg == "--q2") {
            options.q2_model = require_value(i, argc, argv, "--q2");
        } else if (arg == "--pq") {
            options.pq_model = require_value(i, argc, argv, "--pq");
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value(i, argc, argv, "--threads"));
        } else if (arg == "--repetitions") {
            options.repetitions = std::stoi(require_value(i, argc, argv, "--repetitions"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value(i, argc, argv, "--warmup"));
        } else if (arg == "--generations") {
            options.generations = parse_int_list(require_value(i, argc, argv, "--generations"));
        } else if (arg == "--context") {
            options.context = std::stoi(require_value(i, argc, argv, "--context"));
        } else if (arg == "--numa") {
            options.numa = parse_numa(require_value(i, argc, argv, "--numa"));
        } else if (arg == "--output") {
            options.output = require_value(i, argc, argv, "--output");
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.fp16_model.empty() || options.q2_model.empty() || options.pq_model.empty()) {
        throw std::runtime_error("--fp16, --q2, and --pq are required");
    }
    if (options.threads <= 0 || options.repetitions <= 0 || options.warmup < 0 || options.context <= 0) {
        throw std::runtime_error("threads, repetitions, and context must be positive; warmup cannot be negative");
    }
    return options;
}

static void check_model(const std::string & path, const char * label) {
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error(std::string(label) + " does not exist: " + path);
    }
}

static void quiet_log(enum ggml_log_level, const char *, void *) {
}

static double elapsed_seconds(std::chrono::steady_clock::time_point begin,
                              std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

static std::vector<double> run_generation(llama_model * model, const Options & options,
                                           int n_gen, uint64_t seed) {
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = n_gen;
    context_params.n_batch = 2048;
    context_params.n_ubatch = 512;
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads;

    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        throw std::runtime_error("failed to create llama context");
    }

    ggml_threadpool_params threadpool_params = ggml_threadpool_params_default(options.threads);
    threadpool_params.poll = 50;
    ggml_threadpool * threadpool = ggml_threadpool_new(&threadpool_params);
    if (threadpool == nullptr) {
        llama_free(context);
        throw std::runtime_error("failed to create llama.cpp threadpool");
    }
    llama_attach_threadpool(context, threadpool, nullptr);
    llama_set_n_threads(context, options.threads, options.threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int32_t> token_distribution(0, n_vocab - 1);
    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : token_distribution(generator);

    const auto decode = [&](int count) {
        for (int i = 0; i < count; ++i) {
            if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
                llama_free(context);
                ggml_threadpool_free(threadpool);
                throw std::runtime_error("llama_decode failed");
            }
            llama_synchronize(context);
            token = token_distribution(generator);
        }
    };

    for (int i = 0; i < options.warmup; ++i) {
        decode(1);
        llama_memory_clear(llama_get_memory(context), false);
        token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : token_distribution(generator);
    }

    std::vector<double> samples;
    samples.reserve(options.repetitions);
    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        llama_memory_clear(llama_get_memory(context), false);
        const auto begin = std::chrono::steady_clock::now();
        decode(n_gen);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(n_gen / elapsed_seconds(begin, end));
    }

    llama_free(context);
    ggml_threadpool_free(threadpool);
    return samples;
}

static std::vector<Measurement> evaluate_model(const std::string & name, const std::string & path,
                                               bool pq, const Options & options) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    model_params.pq_decode = pq;
    model_params.pq_ds = 4;
    model_params.pq_mode = 0;

    std::cout << "[AE] loading " << name << ": " << path << '\n' << std::flush;
    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model: " + path);
    }
    std::cout << "[AE] loaded " << name << '\n' << std::flush;

    std::vector<Measurement> results;
    for (const int n_gen : options.generations) {
        const std::vector<double> samples = run_generation(model, options, n_gen, 0x4c4c4d41ULL);
        for (size_t repetition = 0; repetition < samples.size(); ++repetition) {
            std::cout << "[AE] " << name << " tg" << n_gen << " run " << (repetition + 1)
                      << ": " << std::fixed << std::setprecision(2) << samples[repetition]
                      << " tokens/s\n" << std::flush;
        }
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double variance = 0.0;
        for (const double value : samples) {
            const double delta = value - mean;
            variance += delta * delta;
        }
        const double stdev = samples.size() > 1 ? std::sqrt(variance / (samples.size() - 1)) : 0.0;
        results.push_back({name, path, n_gen, mean, stdev});
    }
    llama_model_free(model);
    return results;
}

static void write_csv(const std::string & path, const std::vector<Measurement> & results) {
    std::ostream * stream = &std::cout;
    std::ofstream file;
    if (!path.empty()) {
        file.open(path);
        if (!file) {
            throw std::runtime_error("cannot open output: " + path);
        }
        stream = &file;
    }
    *stream << "model,path,generation,tokens_per_second,stdev\n";
    *stream << std::fixed << std::setprecision(2);
    for (const Measurement & result : results) {
        *stream << result.name << ',' << result.path << ',' << result.generation << ','
                << result.mean << ',' << result.stdev << '\n';
    }
}

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        check_model(options.fp16_model, "F16 model");
        check_model(options.q2_model, "Q2 model");
        check_model(options.pq_model, "PQ model");

        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
        llama_numa_init(options.numa);

        const std::vector<std::pair<std::string, std::pair<std::string, bool>>> models = {
            {"F16", {options.fp16_model, false}},
            {"Q2_K", {options.q2_model, false}},
            {"PQ-4c8b", {options.pq_model, true}},
        };
        std::vector<Measurement> results;
        for (const auto & entry : models) {
            const std::vector<Measurement> model_results = evaluate_model(
                entry.first, entry.second.first, entry.second.second, options);
            results.insert(results.end(), model_results.begin(), model_results.end());
        }
        write_csv(options.output, results);
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << "llama_pq: error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
