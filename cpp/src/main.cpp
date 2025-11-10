#include "config.hpp"
#include "dataset.hpp"
#include "model.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::string model = "pico";
    std::string dataset;
    int batch_size = 8;
    int steps = 10;
};

Arguments parse_args(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        std::string token = argv[i];
        auto next_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for flag " + flag);
            }
            return argv[++i];
        };
        if (token == "--model") {
            args.model = next_value(token);
        } else if (token == "--data") {
            args.dataset = next_value(token);
        } else if (token == "--batch-size") {
            args.batch_size = std::stoi(next_value(token));
        } else if (token == "--steps") {
            args.steps = std::stoi(next_value(token));
        } else if (token == "--help") {
            std::cout << "Usage: transformer_train --data <file> [--model <preset>] [--batch-size <n>] [--steps <n>]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + token);
        }
    }
    if (args.dataset.empty()) {
        throw std::runtime_error("--data must point to a text file");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        auto config = preset_from_name(args.model);

        auto text = read_file(args.dataset);
        CharacterTokenizer tokenizer(text);
        auto tokens = tokenizer.encode(text);
        if (tokenizer.vocab_size() != config.vocab_size) {
            config.vocab_size = tokenizer.vocab_size();
        }

        TextDataset dataset(std::move(tokens), config.seq_len);
        TransformerLM model(config);

        std::mt19937 rng(42);
        for (int step = 0; step < args.steps; ++step) {
            auto batch = dataset.sample(rng, args.batch_size);
            const auto& logits = model.forward(batch.input, batch.batch_size, batch.seq_len);
            auto loss = model.loss(batch.target);
            std::cout << "step " << std::setw(4) << step << " loss=" << std::fixed << std::setprecision(4) << loss
                      << " (logits shape=" << logits.shape()[0] << "x" << logits.shape()[1] << "x" << logits.shape()[2]
                      << ")\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}

