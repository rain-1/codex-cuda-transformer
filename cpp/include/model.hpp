#pragma once

#include "config.hpp"

#include <torch/torch.h>

#include <vector>

struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(std::size_t dim, double eps = 1e-5);
    torch::Tensor forward(const torch::Tensor& x);

  private:
    torch::Tensor weight_;
    double eps_;
};
TORCH_MODULE(RMSNorm);

struct MultiHeadAttentionImpl : torch::nn::Module {
    explicit MultiHeadAttentionImpl(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& x);

  private:
    ModelConfig config_;
    torch::nn::Linear qkv_{nullptr};
    torch::nn::Linear proj_{nullptr};
    torch::Tensor mask_;
};
TORCH_MODULE(MultiHeadAttention);

struct FeedForwardImpl : torch::nn::Module {
    explicit FeedForwardImpl(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& x);

  private:
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    double dropout_;
};
TORCH_MODULE(FeedForward);

struct TransformerBlockImpl : torch::nn::Module {
    explicit TransformerBlockImpl(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& x);

  private:
    ModelConfig config_;
    RMSNorm norm1_{nullptr};
    MultiHeadAttention attn_{nullptr};
    RMSNorm norm2_{nullptr};
    FeedForward ff_{nullptr};
};
TORCH_MODULE(TransformerBlock);

struct TransformerLMImpl : torch::nn::Module {
    explicit TransformerLMImpl(const ModelConfig& config);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& idx, const torch::Tensor& targets);

  private:
    ModelConfig config_;
    torch::nn::Embedding token_embedding_{nullptr};
    std::vector<TransformerBlock> blocks_;
    RMSNorm norm_{nullptr};
    torch::nn::Linear head_{nullptr};
};
TORCH_MODULE(TransformerLM);

