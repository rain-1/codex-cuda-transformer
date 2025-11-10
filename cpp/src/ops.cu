#include "ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void fill_kernel(float* data, std::size_t size, float value) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void add_inplace_kernel(float* dst, const float* src, std::size_t size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}

__global__ void copy_kernel(float* dst, const float* src, std::size_t size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void embedding_forward_kernel(const float* weight, const int* indices, float* output, int rows, int dim) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(rows) * dim;
    if (idx >= total) {
        return;
    }
    int row = idx / dim;
    int col = idx % dim;
    int token = indices[row];
    output[idx] = weight[static_cast<std::size_t>(token) * dim + col];
}

int rows_from_shape(const std::vector<int>& shape) {
    if (shape.empty()) {
        return 0;
    }
    int rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
        rows *= shape[i];
    }
    return rows;
}

__global__ void linear_forward_kernel(const float* input, const float* weight, const float* bias, float* output,
                                      int rows, int in_dim, int out_dim) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(rows) * out_dim;
    if (idx >= total) {
        return;
    }
    int row = idx / out_dim;
    int col = idx % out_dim;
    const float* input_row = input + static_cast<std::size_t>(row) * in_dim;
    const float* weight_row = weight + static_cast<std::size_t>(col) * in_dim;
    float sum = bias ? bias[col] : 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        sum += input_row[i] * weight_row[i];
    }
    output[idx] = sum;
}

constexpr float kGeluAlpha = 0.7978845608028654f;

__device__ float gelu_value(float x) {
    return 0.5f * x * (1.0f + tanhf(kGeluAlpha * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_forward_kernel(const float* input, float* output, std::size_t size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_value(input[idx]);
    }
}

__global__ void rmsnorm_forward_kernel(const float* input, const float* weight, float* output, float* inv_rms,
                                       float* norm_cache, int rows, int dim, float eps) {
    auto row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* input_row = input + static_cast<std::size_t>(row) * dim;
    float* output_row = output + static_cast<std::size_t>(row) * dim;
    float* norm_row = norm_cache + static_cast<std::size_t>(row) * dim;

    float mean_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float v = input_row[i];
        mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(dim);
    float inv = rsqrtf(mean_sq + eps);
    inv_rms[row] = inv;
    for (int i = 0; i < dim; ++i) {
        float norm = input_row[i] * inv;
        norm_row[i] = norm;
        output_row[i] = norm * weight[i];
    }
}

__global__ void softmax_forward_kernel(float* logits, int rows, int dim) {
    auto row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float* row_ptr = logits + static_cast<std::size_t>(row) * dim;
    float max_val = row_ptr[0];
    for (int i = 1; i < dim; ++i) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        row_ptr[i] = expf(row_ptr[i] - max_val);
        sum += row_ptr[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; ++i) {
        row_ptr[i] *= inv_sum;
    }
}

__global__ void attention_scores_kernel(const float* q, const float* k, float* scores, int batch, int heads,
                                        int seq_len, int head_dim, float scale) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * seq_len;
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int j = tmp % seq_len;
    tmp /= seq_len;
    int i = tmp % seq_len;
    tmp /= seq_len;
    int head = tmp % heads;
    int b = tmp / heads;

    const float* q_vec = q + (((b * heads + head) * seq_len + i) * head_dim);
    const float* k_vec = k + (((b * heads + head) * seq_len + j) * head_dim);
    float sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        sum += q_vec[d] * k_vec[d];
    }
    scores[idx] = sum * scale;
}

__global__ void attention_mask_future_kernel(float* scores, int batch, int heads, int seq_len) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * seq_len;
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int j = tmp % seq_len;
    tmp /= seq_len;
    int i = tmp % seq_len;
    if (j > i) {
        scores[idx] = -1e9f;
    }
}

__global__ void attention_apply_kernel(const float* scores, const float* v, float* output, int batch, int heads,
                                       int seq_len, int head_dim) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * head_dim;
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int d = tmp % head_dim;
    tmp /= head_dim;
    int i = tmp % seq_len;
    tmp /= seq_len;
    int head = tmp % heads;
    int b = tmp / heads;

    const float* scores_row = scores + (((b * heads + head) * seq_len + i) * seq_len);
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float coeff = scores_row[j];
        const float* v_vec = v + (((b * heads + head) * seq_len + j) * head_dim);
        sum += coeff * v_vec[d];
    }
    output[idx] = sum;
}

__global__ void split_qkv_kernel(const float* qkv, float* q, float* k, float* v, int batch, int seq_len, int heads,
                                 int head_dim) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * head_dim;
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int d = tmp % head_dim;
    tmp /= head_dim;
    int seq = tmp % seq_len;
    tmp /= seq_len;
    int head = tmp % heads;
    int b = tmp / heads;

    int model_dim = heads * head_dim;
    std::size_t row_offset = (static_cast<std::size_t>(b) * seq_len + seq) * (3 * model_dim);
    int head_offset = head * head_dim + d;
    q[idx] = qkv[row_offset + head_offset];
    k[idx] = qkv[row_offset + model_dim + head_offset];
    v[idx] = qkv[row_offset + 2 * model_dim + head_offset];
}

__global__ void combine_heads_kernel(const float* src, float* dst, int batch, int heads, int seq_len, int head_dim) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto total = static_cast<std::size_t>(batch) * seq_len * heads * head_dim;
    if (idx >= total) {
        return;
    }
    int tmp = idx;
    int d = tmp % head_dim;
    tmp /= head_dim;
    int head = tmp % heads;
    tmp /= heads;
    int seq = tmp % seq_len;
    int b = tmp / seq_len;

    std::size_t src_index = (((b * heads + head) * seq_len + seq) * head_dim) + d;
    int model_dim = heads * head_dim;
    std::size_t dst_index = ((static_cast<std::size_t>(b) * seq_len + seq) * model_dim) + head * head_dim + d;
    dst[dst_index] = src[src_index];
}

}  // namespace

void fill_tensor(Tensor& tensor, float value) {
    auto size = tensor.size();
    auto blocks = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
    fill_kernel<<<blocks, kBlockSize>>>(tensor.data(), size, value);
    check_cuda(cudaGetLastError(), "fill_kernel");
}

void add_inplace(Tensor& dst, const Tensor& src) {
    auto size = dst.size();
    auto blocks = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
    add_inplace_kernel<<<blocks, kBlockSize>>>(dst.data(), src.data(), size);
    check_cuda(cudaGetLastError(), "add_inplace_kernel");
}

void copy_tensor(const Tensor& src, Tensor& dst) {
    if (dst.size() != src.size()) {
        dst.allocate(src.shape());
    }
    auto size = src.size();
    auto blocks = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
    copy_kernel<<<blocks, kBlockSize>>>(dst.data(), src.data(), size);
    check_cuda(cudaGetLastError(), "copy_kernel");
}

void embedding_forward(const Tensor& weight, const std::vector<int>& indices, int batch, int seq_len, Tensor& output) {
    int rows = batch * seq_len;
    auto shape = std::vector<int>{batch, seq_len, static_cast<int>(weight.shape().back())};
    output.allocate(shape);
    int* device_indices = nullptr;
    check_cuda(cudaMalloc(&device_indices, rows * sizeof(int)), "cudaMalloc embedding indices");
    check_cuda(cudaMemcpy(device_indices, indices.data(), rows * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy embedding indices");
    auto total = static_cast<std::size_t>(rows) * weight.shape().back();
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    embedding_forward_kernel<<<blocks, kBlockSize>>>(weight.data(), device_indices, output.data(), rows,
                                                     weight.shape().back());
    check_cuda(cudaGetLastError(), "embedding_forward_kernel");
    check_cuda(cudaFree(device_indices), "cudaFree embedding indices");
}

void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    const auto& shape = input.shape();
    int in_dim = shape.back();
    int out_dim = weight.shape().front();
    auto out_shape = shape;
    out_shape.back() = out_dim;
    output.allocate(out_shape);

    int rows = rows_from_shape(shape);
    auto total = static_cast<std::size_t>(rows) * out_dim;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    linear_forward_kernel<<<blocks, kBlockSize>>>(input.data(), weight.data(), bias.size() ? bias.data() : nullptr,
                                                  output.data(), rows, in_dim, out_dim);
    check_cuda(cudaGetLastError(), "linear_forward_kernel");
}

void gelu_forward(const Tensor& input, Tensor& output) {
    output.allocate(input.shape());
    auto size = input.size();
    auto blocks = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
    gelu_forward_kernel<<<blocks, kBlockSize>>>(input.data(), output.data(), size);
    check_cuda(cudaGetLastError(), "gelu_forward_kernel");
}

void rmsnorm_forward(const Tensor& input, const Tensor& weight, float eps, Tensor& output, Tensor& inv_rms,
                     Tensor& norm_cache) {
    const auto& shape = input.shape();
    int dim = shape.back();
    int rows = rows_from_shape(shape);
    output.allocate(shape);
    inv_rms.allocate({rows});
    norm_cache.allocate(shape);
    rmsnorm_forward_kernel<<<rows, 1>>>(input.data(), weight.data(), output.data(), inv_rms.data(), norm_cache.data(), rows,
                                        dim, eps);
    check_cuda(cudaGetLastError(), "rmsnorm_forward_kernel");
}

void softmax_forward(Tensor& logits) {
    const auto& shape = logits.shape();
    int dim = shape.back();
    int rows = rows_from_shape(shape);
    softmax_forward_kernel<<<rows, 1>>>(logits.data(), rows, dim);
    check_cuda(cudaGetLastError(), "softmax_forward_kernel");
}

void attention_scores(const Tensor& q, const Tensor& k, Tensor& scores, float scale) {
    const auto& shape = q.shape();
    int batch = shape[0];
    int heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    scores.allocate({batch, heads, seq_len, seq_len});
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * seq_len;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    attention_scores_kernel<<<blocks, kBlockSize>>>(q.data(), k.data(), scores.data(), batch, heads, seq_len, head_dim, scale);
    check_cuda(cudaGetLastError(), "attention_scores_kernel");
}

void attention_mask_future(Tensor& scores) {
    const auto& shape = scores.shape();
    int batch = shape[0];
    int heads = shape[1];
    int seq_len = shape[2];
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * seq_len;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    attention_mask_future_kernel<<<blocks, kBlockSize>>>(scores.data(), batch, heads, seq_len);
    check_cuda(cudaGetLastError(), "attention_mask_future_kernel");
}

void attention_apply(const Tensor& scores, const Tensor& v, Tensor& output) {
    const auto& shape = scores.shape();
    int batch = shape[0];
    int heads = shape[1];
    int seq_len = shape[2];
    int head_dim = v.shape().back();
    output.allocate({batch, heads, seq_len, head_dim});
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * head_dim;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    attention_apply_kernel<<<blocks, kBlockSize>>>(scores.data(), v.data(), output.data(), batch, heads, seq_len, head_dim);
    check_cuda(cudaGetLastError(), "attention_apply_kernel");
}

void split_qkv(const Tensor& qkv, int heads, Tensor& q, Tensor& k, Tensor& v) {
    const auto& shape = qkv.shape();
    int batch = shape[0];
    int seq_len = shape[1];
    int model_dim = shape[2] / 3;
    int head_dim = model_dim / heads;
    q.allocate({batch, heads, seq_len, head_dim});
    k.allocate({batch, heads, seq_len, head_dim});
    v.allocate({batch, heads, seq_len, head_dim});
    auto total = static_cast<std::size_t>(batch) * heads * seq_len * head_dim;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    split_qkv_kernel<<<blocks, kBlockSize>>>(qkv.data(), q.data(), k.data(), v.data(), batch, seq_len, heads, head_dim);
    check_cuda(cudaGetLastError(), "split_qkv_kernel");
}

void combine_heads(const Tensor& src, Tensor& dst) {
    const auto& shape = src.shape();
    int batch = shape[0];
    int heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    int model_dim = heads * head_dim;
    dst.allocate({batch, seq_len, model_dim});
    auto total = static_cast<std::size_t>(batch) * seq_len * model_dim;
    auto blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    combine_heads_kernel<<<blocks, kBlockSize>>>(src.data(), dst.data(), batch, heads, seq_len, head_dim);
    check_cuda(cudaGetLastError(), "combine_heads_kernel");
}

float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets) {
    Tensor probs(logits.shape());
    copy_tensor(logits, probs);
    softmax_forward(probs);
    auto host_probs = probs.to_host();
    const auto& shape = logits.shape();
    int dim = shape.back();
    int rows = rows_from_shape(shape);
    if (static_cast<int>(targets.size()) != rows) {
        throw std::runtime_error("Target size mismatch for cross entropy");
    }
    double loss = 0.0;
    for (int row = 0; row < rows; ++row) {
        int target = targets[row];
        float prob = host_probs[static_cast<std::size_t>(row) * dim + target];
        loss -= std::log(std::max(prob, 1e-9f));
    }
    return static_cast<float>(loss / rows);
}

