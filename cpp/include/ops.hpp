#pragma once

#include "tensor.hpp"

#include <cstddef>
#include <vector>

void fill_tensor(Tensor& tensor, float value);
void add_inplace(Tensor& dst, const Tensor& src);
void copy_tensor(const Tensor& src, Tensor& dst);

void embedding_forward(const Tensor& weight, const std::vector<int>& indices, int batch, int seq_len, Tensor& output);

void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);
void gelu_forward(const Tensor& input, Tensor& output);
void rmsnorm_forward(const Tensor& input, const Tensor& weight, float eps, Tensor& output, Tensor& inv_rms,
                     Tensor& norm_cache);
void softmax_forward(Tensor& logits);

void attention_scores(const Tensor& q, const Tensor& k, Tensor& scores, float scale);
void attention_mask_future(Tensor& scores);
void attention_apply(const Tensor& scores, const Tensor& v, Tensor& output);
void split_qkv(const Tensor& qkv, int heads, Tensor& q, Tensor& k, Tensor& v);
void combine_heads(const Tensor& src, Tensor& dst);

float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets);

