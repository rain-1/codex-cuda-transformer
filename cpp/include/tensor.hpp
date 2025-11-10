#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape) { allocate(shape); }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept { move_from(std::move(other)); }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }
    ~Tensor() { release(); }

    void allocate(const std::vector<int>& shape) {
        release();
        shape_ = shape;
        size_ = 1;
        for (int dim : shape_) {
            size_ *= dim;
        }
        if (size_ == 0) {
            data_ = nullptr;
            return;
        }
        check_cuda(cudaMalloc(&data_, size_ * sizeof(float)), "cudaMalloc tensor");
    }

    void resize_like(const Tensor& other) { allocate(other.shape_); }

    float* data() { return data_; }
    const float* data() const { return data_; }

    const std::vector<int>& shape() const { return shape_; }
    std::size_t size() const { return size_; }

    std::vector<float> to_host() const {
        std::vector<float> host(size_);
        if (size_ > 0) {
            check_cuda(cudaMemcpy(host.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy tensor to host");
        }
        return host;
    }

    void from_host(const std::vector<float>& host) {
        if (host.size() != size_) {
            throw std::runtime_error("Host data size does not match tensor");
        }
        if (size_ > 0) {
            check_cuda(cudaMemcpy(data_, host.data(), size_ * sizeof(float), cudaMemcpyHostToDevice),
                       "cudaMemcpy tensor from host");
        }
    }

    void zero() {
        if (size_ > 0) {
            check_cuda(cudaMemset(data_, 0, size_ * sizeof(float)), "cudaMemset tensor zero");
        }
    }

  private:
    void move_from(Tensor&& other) {
        data_ = other.data_;
        size_ = other.size_;
        shape_ = std::move(other.shape_);
        other.data_ = nullptr;
        other.size_ = 0;
        other.shape_.clear();
    }

    void release() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
        shape_.clear();
    }

    float* data_ = nullptr;
    std::size_t size_ = 0;
    std::vector<int> shape_;
};

