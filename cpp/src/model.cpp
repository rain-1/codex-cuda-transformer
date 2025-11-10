#include "model.hpp"

#include <cmath>
#include <string>

#include <torch/indexing.h>
#include <torch/nn/functional.h>

RMSNormImpl::RMSNormImpl(std::size_t dim, double eps) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({static_cast<long>(dim)}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
    auto variance = x.pow(2).mean(-1, true);
    auto normed = x * torch::rsqrt(variance + eps_);
    return normed * weight_;
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(const ModelConfig& config) : config_(config) {
    qkv_ = register_module("qkv", torch::nn::Linear(config.d_model, config.d_model * 3, false));
    proj_ = register_module("proj", torch::nn::Linear(config.d_model, config.d_model, false));
    mask_ = torch::tril(torch::ones({static_cast<long>(config.seq_len), static_cast<long>(config.seq_len)}));
}

torch::Tensor MultiHeadAttentionImpl::forward(const torch::Tensor& x) {
    auto batch = x.size(0);
    auto seq_len = x.size(1);
    auto head_dim = config_.d_model / config_.n_heads;

    auto qkv = qkv_->forward(x).view({batch, seq_len, 3, static_cast<long>(config_.n_heads), static_cast<long>(head_dim)});
    qkv = qkv.permute({2, 0, 3, 1, 4});
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    auto mask = mask_.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len)});
    scores = scores.masked_fill(mask == 0, -1e9);
    auto attn = torch::softmax(scores, -1);
    auto out = torch::matmul(attn, v);
    out = out.transpose(1, 2).contiguous().view({batch, seq_len, static_cast<long>(config_.d_model)});
    return proj_->forward(out);
}

FeedForwardImpl::FeedForwardImpl(const ModelConfig& config) : dropout_(config.dropout) {
    fc1_ = register_module("fc1", torch::nn::Linear(config.d_model, config.d_ff));
    fc2_ = register_module("fc2", torch::nn::Linear(config.d_ff, config.d_model));
}

torch::Tensor FeedForwardImpl::forward(const torch::Tensor& x) {
    auto out = torch::gelu(fc1_->forward(x));
    out = torch::dropout(out, dropout_, is_training());
    out = fc2_->forward(out);
    return torch::dropout(out, dropout_, is_training());
}

TransformerBlockImpl::TransformerBlockImpl(const ModelConfig& config) : config_(config) {
    norm1_ = register_module("norm1", RMSNorm(config.d_model));
    attn_ = register_module("attn", MultiHeadAttention(config));
    norm2_ = register_module("norm2", RMSNorm(config.d_model));
    ff_ = register_module("ff", FeedForward(config));
}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& x) {
    auto attn_out = attn_->forward(norm1_->forward(x));
    auto residual = x + attn_out;
    auto ff_out = ff_->forward(norm2_->forward(residual));
    return residual + ff_out;
}

TransformerLMImpl::TransformerLMImpl(const ModelConfig& config) : config_(config) {
    token_embedding_ = register_module("token_embedding", torch::nn::Embedding(config.vocab_size, config.d_model));
    norm_ = register_module("norm", RMSNorm(config.d_model));
    head_ = register_module("head", torch::nn::Linear(config.d_model, config.vocab_size, false));
    for (std::size_t i = 0; i < config.n_layers; ++i) {
        auto name = "block_" + std::to_string(i);
        auto block = TransformerBlock(config);
        blocks_.push_back(register_module(name, block));
    }
}

std::pair<torch::Tensor, torch::Tensor> TransformerLMImpl::forward(const torch::Tensor& idx, const torch::Tensor& targets) {
    auto x = token_embedding_->forward(idx);
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    x = norm_->forward(x);
    auto logits = head_->forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits.view({-1, static_cast<long>(config_.vocab_size)}), targets.view({-1}));
    return {logits, loss};
}

