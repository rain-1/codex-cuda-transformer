#pragma once

#include <cstddef>
#include <string>

struct ModelConfig {
    std::size_t vocab_size;
    std::size_t seq_len;
    std::size_t d_model;
    std::size_t n_layers;
    std::size_t n_heads;
    std::size_t d_ff;
    double dropout;
};

struct TrainingConfig {
    std::string model_name;
    std::string dataset_path;
    std::size_t batch_size;
    std::size_t micro_batch_size;
    std::size_t steps;
    double lr;
    double weight_decay;
    double warmup_ratio;
    std::size_t eval_interval;
    std::size_t eval_iters;
    double grad_clip;
    std::string device;
    bool use_wandb;
    std::string wandb_project;
    std::string wandb_run;
};

ModelConfig preset_from_name(const std::string& name);

