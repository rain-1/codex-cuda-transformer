#pragma once

#include "config.hpp"
#include "ops.hpp"

#include <random>
#include <utility>
#include <vector>

class LinearLayer {
  public:
    LinearLayer() = default;
    LinearLayer(int in_dim, int out_dim, bool bias, std::mt19937& rng) { init(in_dim, out_dim, bias, rng); }

    void init(int in_dim, int out_dim, bool bias, std::mt19937& rng);
    void forward(const Tensor& input, Tensor& output) const;

  private:
    Tensor weight_;
    Tensor bias_;
    bool has_bias_ = false;
};

class RMSNormLayer {
  public:
    RMSNormLayer() = default;
    RMSNormLayer(int dim, float eps);

    void forward(const Tensor& input, Tensor& output);

  private:
    Tensor weight_;
    Tensor inv_rms_;
    Tensor norm_cache_;
    float eps_ = 1e-5f;
};

class FeedForward {
  public:
    FeedForward() = default;
    FeedForward(const ModelConfig& config, std::mt19937& rng);

    void forward(const Tensor& input, Tensor& output);

  private:
    LinearLayer fc1_;
    LinearLayer fc2_;
    Tensor hidden_;
    Tensor activated_;
};

class MultiHeadAttention {
  public:
    MultiHeadAttention() = default;
    MultiHeadAttention(const ModelConfig& config, std::mt19937& rng);

    void forward(const Tensor& input, Tensor& output);

  private:
    ModelConfig config_;
    LinearLayer qkv_;
    LinearLayer proj_;
    float scale_ = 1.0f;
    Tensor qkv_proj_;
    Tensor q_;
    Tensor k_;
    Tensor v_;
    Tensor scores_;
    Tensor softmax_;
    Tensor context_;
    Tensor merged_;
};

class TransformerBlock {
  public:
    TransformerBlock() = default;
    TransformerBlock(const ModelConfig& config, std::mt19937& rng);

    void forward(const Tensor& input, Tensor& output);

  private:
    RMSNormLayer norm1_;
    MultiHeadAttention attn_;
    RMSNormLayer norm2_;
    FeedForward ff_;
    Tensor norm1_out_;
    Tensor attn_out_;
    Tensor norm2_out_;
    Tensor ff_out_;
};

class TransformerLM {
  public:
    explicit TransformerLM(const ModelConfig& config);

    const Tensor& forward(const std::vector<int>& tokens, int batch_size, int seq_len);
    float loss(const std::vector<int>& targets) const;

  private:
    ModelConfig config_;
    Tensor embedding_;
    std::vector<TransformerBlock> blocks_;
    RMSNormLayer norm_;
    LinearLayer head_;
    Tensor hidden_;
    Tensor block_output_;
    Tensor norm_out_;
    Tensor logits_;
};

