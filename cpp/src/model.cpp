#include "model.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void init_normal(Tensor& tensor, std::mt19937& rng, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> host(tensor.size());
    for (auto& v : host) {
        v = dist(rng);
    }
    tensor.from_host(host);
}

void init_constant(Tensor& tensor, float value) {
    std::vector<float> host(tensor.size(), value);
    tensor.from_host(host);
}

}  // namespace

void LinearLayer::init(int in_dim, int out_dim, bool bias, std::mt19937& rng) {
    has_bias_ = bias;
    weight_.allocate({out_dim, in_dim});
    float stddev = 1.0f / std::sqrt(static_cast<float>(in_dim));
    init_normal(weight_, rng, stddev);
    if (bias) {
        bias_.allocate({out_dim});
        init_constant(bias_, 0.0f);
    }
}

void LinearLayer::forward(const Tensor& input, Tensor& output) const {
    linear_forward(input, weight_, bias_, output);
}

RMSNormLayer::RMSNormLayer(int dim, float eps) : eps_(eps) {
    weight_.allocate({dim});
    init_constant(weight_, 1.0f);
}

void RMSNormLayer::forward(const Tensor& input, Tensor& output) {
    rmsnorm_forward(input, weight_, eps_, output, inv_rms_, norm_cache_);
}

FeedForward::FeedForward(const ModelConfig& config, std::mt19937& rng) {
    fc1_.init(static_cast<int>(config.d_model), static_cast<int>(config.d_ff), true, rng);
    fc2_.init(static_cast<int>(config.d_ff), static_cast<int>(config.d_model), true, rng);
}

void FeedForward::forward(const Tensor& input, Tensor& output) {
    fc1_.forward(input, hidden_);
    gelu_forward(hidden_, activated_);
    fc2_.forward(activated_, output);
}

MultiHeadAttention::MultiHeadAttention(const ModelConfig& config, std::mt19937& rng) : config_(config) {
    if (config.d_model % config.n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }
    qkv_.init(static_cast<int>(config.d_model), static_cast<int>(config.d_model * 3), false, rng);
    proj_.init(static_cast<int>(config.d_model), static_cast<int>(config.d_model), false, rng);
    scale_ = 1.0f / std::sqrt(static_cast<float>(config.d_model / config.n_heads));
}

void MultiHeadAttention::forward(const Tensor& input, Tensor& output) {
    qkv_.forward(input, qkv_proj_);
    split_qkv(qkv_proj_, static_cast<int>(config_.n_heads), q_, k_, v_);
    attention_scores(q_, k_, scores_, scale_);
    attention_mask_future(scores_);
    copy_tensor(scores_, softmax_);
    softmax_forward(softmax_);
    attention_apply(softmax_, v_, context_);
    combine_heads(context_, merged_);
    proj_.forward(merged_, output);
}

TransformerBlock::TransformerBlock(const ModelConfig& config, std::mt19937& rng)
    : norm1_(static_cast<int>(config.d_model), 1e-5f),
      attn_(config, rng),
      norm2_(static_cast<int>(config.d_model), 1e-5f),
      ff_(config, rng) {}

void TransformerBlock::forward(const Tensor& input, Tensor& output) {
    norm1_.forward(input, norm1_out_);
    attn_.forward(norm1_out_, attn_out_);
    copy_tensor(input, output);
    add_inplace(output, attn_out_);
    norm2_.forward(output, norm2_out_);
    ff_.forward(norm2_out_, ff_out_);
    add_inplace(output, ff_out_);
}

TransformerLM::TransformerLM(const ModelConfig& config) : config_(config) {
    if (config_.d_model % config_.n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }
    std::random_device rd;
    std::mt19937 rng(rd());

    embedding_.allocate({static_cast<int>(config_.vocab_size), static_cast<int>(config_.d_model)});
    float stddev = 1.0f / std::sqrt(static_cast<float>(config_.d_model));
    init_normal(embedding_, rng, stddev);

    blocks_.reserve(config_.n_layers);
    for (std::size_t i = 0; i < config_.n_layers; ++i) {
        blocks_.emplace_back(config_, rng);
    }
    norm_ = RMSNormLayer(static_cast<int>(config_.d_model), 1e-5f);
    head_.init(static_cast<int>(config_.d_model), static_cast<int>(config_.vocab_size), false, rng);
}

const Tensor& TransformerLM::forward(const std::vector<int>& tokens, int batch_size, int seq_len) {
    if (static_cast<int>(tokens.size()) != batch_size * seq_len) {
        throw std::runtime_error("Token batch has unexpected size");
    }
    embedding_forward(embedding_, tokens, batch_size, seq_len, hidden_);

    Tensor* current = &hidden_;
    Tensor* next = &block_output_;
    for (auto& block : blocks_) {
        block.forward(*current, *next);
        std::swap(current, next);
    }
    norm_.forward(*current, norm_out_);
    head_.forward(norm_out_, logits_);
    return logits_;
}

float TransformerLM::loss(const std::vector<int>& targets) const {
    return cross_entropy_loss(logits_, targets);
}

