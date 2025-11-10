#include "config.hpp"
#include "dataset.hpp"
#include "model.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/indexing.h>

namespace {

struct Arguments {
    std::string model = "pico";
    std::string dataset;
    std::size_t batch_size = 64;
    std::size_t micro_batch_size = 8;
    std::size_t steps = 1000;
    double lr = 3e-4;
    double weight_decay = 0.1;
    double warmup_ratio = 0.03;
    std::size_t eval_interval = 100;
    std::size_t eval_iters = 10;
    double grad_clip = 1.0;
    std::string device = torch::cuda::is_available() ? "cuda" : "cpu";
    bool wandb = false;
    std::string wandb_project = "codex-transformer";
    std::string wandb_run;
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
            args.batch_size = std::stoul(next_value(token));
        } else if (token == "--micro-batch-size") {
            args.micro_batch_size = std::stoul(next_value(token));
        } else if (token == "--steps") {
            args.steps = std::stoul(next_value(token));
        } else if (token == "--lr") {
            args.lr = std::stod(next_value(token));
        } else if (token == "--weight-decay") {
            args.weight_decay = std::stod(next_value(token));
        } else if (token == "--warmup-ratio") {
            args.warmup_ratio = std::stod(next_value(token));
        } else if (token == "--eval-interval") {
            args.eval_interval = std::stoul(next_value(token));
        } else if (token == "--eval-iters") {
            args.eval_iters = std::stoul(next_value(token));
        } else if (token == "--grad-clip") {
            args.grad_clip = std::stod(next_value(token));
        } else if (token == "--device") {
            args.device = next_value(token);
        } else if (token == "--wandb") {
            args.wandb = true;
        } else if (token == "--wandb-project") {
            args.wandb_project = next_value(token);
        } else if (token == "--wandb-run") {
            args.wandb_run = next_value(token);
        } else if (token == "--help") {
            std::cout << "Usage: train --data <path> [options]\n";
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

struct WandbLogger {
    bool enabled;
    std::FILE* pipe = nullptr;

    WandbLogger(bool use, const std::string& project, const std::string& run_name)
        : enabled(use) {
        if (!enabled) {
            return;
        }
        std::ostringstream cmd;
        cmd << "python3 -m codex_lm.wandb_stream --project " << std::quoted(project);
        if (!run_name.empty()) {
            cmd << " --name " << std::quoted(run_name);
        }
        pipe = popen(cmd.str().c_str(), "w");
        if (!pipe) {
            throw std::runtime_error("Failed to start wandb stream. Ensure python dependencies are installed.");
        }
    }

    ~WandbLogger() {
        if (pipe) {
            pclose(pipe);
        }
    }

    void log(std::size_t step, const std::string& payload) {
        if (!enabled || !pipe) {
            return;
        }
        std::ostringstream line;
        line << "{";
        if (step != std::numeric_limits<std::size_t>::max()) {
            line << "\"step\":" << step;
            if (!payload.empty()) {
                line << ",";
            }
        }
        line << payload << "}\n";
        auto str = line.str();
        std::fwrite(str.data(), 1, str.size(), pipe);
        std::fflush(pipe);
    }
};

std::string metric_payload(const std::vector<std::pair<std::string, double>>& metrics) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        ss << "\"" << metrics[i].first << "\":" << metrics[i].second;
        if (i + 1 < metrics.size()) {
            ss << ",";
        }
    }
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    auto config = preset_from_name(args.model);

    constexpr double PI = 3.14159265358979323846;

    auto text = read_file(args.dataset);
    CharacterTokenizer tokenizer(text);
    auto tokens_vec = tokenizer.encode(text);
    auto tokens = torch::tensor(tokens_vec, torch::kInt64);

    if (tokenizer.vocab_size() != config.vocab_size) {
        config.vocab_size = tokenizer.vocab_size();
    }

    auto split_idx = static_cast<std::size_t>(tokens.size(0) * 0.9);
    auto train_tokens = tokens.index({torch::indexing::Slice(0, split_idx)});
    auto val_tokens = tokens.index({torch::indexing::Slice(split_idx, tokens.size(0))});

    auto train_dataset = TextDataset(train_tokens, config.seq_len).map(torch::data::transforms::Stack<>());
    auto val_dataset = TextDataset(val_tokens, config.seq_len).map(torch::data::transforms::Stack<>());

    auto accum_steps = std::max<std::size_t>(1, args.batch_size / args.micro_batch_size);
    auto options = torch::data::DataLoaderOptions(args.micro_batch_size).drop_last(true).workers(2);
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(train_dataset, options);
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(val_dataset, options);

    torch::Device device(args.device == "cuda" && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    TransformerLM model(config);
    model->to(device);

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(args.lr).weight_decay(args.weight_decay).betas({0.9, 0.95}));
    auto total_steps = static_cast<long>(args.steps);
    auto warmup_iters = static_cast<long>(total_steps * args.warmup_ratio);
    auto scheduler = torch::optim::LambdaLR(optimizer, [=](long step) {
        if (step < warmup_iters) {
            return std::max(1e-6, static_cast<double>(step + 1) / std::max<long>(1, warmup_iters));
        }
        auto progress = static_cast<double>(step - warmup_iters) / std::max<long>(1, total_steps - warmup_iters);
        return 0.5 * (1.0 + std::cos(PI * progress));
    });

    WandbLogger logger(args.wandb, args.wandb_project, args.wandb_run);

    auto step = 0UL;
    auto train_iter = train_loader->begin();
    for (; step < args.steps; ++step) {
        auto start = std::chrono::high_resolution_clock::now();
        model->train();
        double loss_value = 0.0;
        optimizer.zero_grad();
        for (std::size_t accum = 0; accum < accum_steps; ++accum) {
            if (train_iter == train_loader->end()) {
                train_iter = train_loader->begin();
            }
            auto batch = *train_iter;
            ++train_iter;
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model->forward(data, target);
            auto loss = output.second / static_cast<double>(accum_steps);
            loss.backward();
            loss_value += loss.item<double>();
        }
        if (args.grad_clip > 0) {
            torch::nn::utils::clip_grad_norm_(model->parameters(), args.grad_clip);
        }
        optimizer.step();
        scheduler.step();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();

        auto current_lr = static_cast<const torch::optim::AdamWOptions&>(optimizer.param_groups()[0].options()).lr();
        auto payload = metric_payload({{"train/loss", loss_value}, {"lr", current_lr}, {"step_time", duration}});
        logger.log(step + 1, payload);
        std::cout << "step " << (step + 1) << " loss=" << loss_value << " lr=" << current_lr << " time=" << duration << "s\n";

        if ((step + 1) % args.eval_interval == 0) {
            model->eval();
            double val_loss = 0.0;
            std::size_t count = 0;
            for (auto& batch : *val_loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                auto output = model->forward(data, target);
                val_loss += output.second.item<double>();
                if (++count >= args.eval_iters) {
                    break;
                }
            }
            val_loss /= std::max<std::size_t>(1, count);
            auto eval_payload = metric_payload({{"val/loss", val_loss}, {"val/perplexity", std::exp(val_loss)}});
            logger.log(step + 1, eval_payload);
            std::cout << "val loss=" << val_loss << " ppl=" << std::exp(val_loss) << "\n";
        }
    }

    torch::save(model->state_dict(), "cpp_last_model.pt");
    return 0;
}

