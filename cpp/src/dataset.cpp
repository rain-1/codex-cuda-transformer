#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

CharacterTokenizer::CharacterTokenizer(const std::string& text) {
    std::unordered_set<char> chars(text.begin(), text.end());
    itos_.assign(chars.begin(), chars.end());
    std::sort(itos_.begin(), itos_.end());
    for (std::size_t i = 0; i < itos_.size(); ++i) {
        stoi_[itos_[i]] = static_cast<int>(i);
    }
}

std::vector<int> CharacterTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char ch : text) {
        auto it = stoi_.find(ch);
        if (it == stoi_.end()) {
            throw std::runtime_error("Character not in vocabulary");
        }
        tokens.push_back(it->second);
    }
    return tokens;
}

std::string CharacterTokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    out.reserve(tokens.size());
    for (int token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= itos_.size()) {
            throw std::runtime_error("Token out of range");
        }
        out.push_back(itos_[token]);
    }
    return out;
}

TextDataset::TextDataset(std::vector<int> tokens, std::size_t seq_len)
    : tokens_(std::move(tokens)), seq_len_(seq_len) {
    if (tokens_.size() < seq_len_ + 1) {
        throw std::runtime_error("Dataset too small for requested sequence length");
    }
}

Batch TextDataset::sample(std::mt19937& rng, int batch_size) const {
    std::uniform_int_distribution<std::size_t> dist(0, tokens_.size() - seq_len_ - 1);
    Batch batch;
    batch.batch_size = batch_size;
    batch.seq_len = static_cast<int>(seq_len_);
    batch.input.resize(batch_size * batch.seq_len);
    batch.target.resize(batch_size * batch.seq_len);
    for (int b = 0; b < batch_size; ++b) {
        std::size_t start = dist(rng);
        for (std::size_t i = 0; i < seq_len_; ++i) {
            batch.input[b * batch.seq_len + i] = tokens_[start + i];
            batch.target[b * batch.seq_len + i] = tokens_[start + i + 1];
        }
    }
    return batch;
}

Batch TextDataset::sample_sequential(std::size_t offset, int batch_size) const {
    Batch batch;
    batch.batch_size = batch_size;
    batch.seq_len = static_cast<int>(seq_len_);
    batch.input.resize(batch_size * batch.seq_len);
    batch.target.resize(batch_size * batch.seq_len);
    for (int b = 0; b < batch_size; ++b) {
        std::size_t start = offset + static_cast<std::size_t>(b) * seq_len_;
        if (start + seq_len_ + 1 > tokens_.size()) {
            break;
        }
        for (std::size_t i = 0; i < seq_len_; ++i) {
            batch.input[b * batch.seq_len + i] = tokens_[start + i];
            batch.target[b * batch.seq_len + i] = tokens_[start + i + 1];
        }
    }
    return batch;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

