#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

CharacterTokenizer::CharacterTokenizer(const std::string& text) {
    std::unordered_set<char> chars(text.begin(), text.end());
    itos_.assign(chars.begin(), chars.end());
    std::sort(itos_.begin(), itos_.end());
    for (std::size_t i = 0; i < itos_.size(); ++i) {
        stoi_[itos_[i]] = static_cast<int64_t>(i);
    }
}

std::vector<int64_t> CharacterTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> tokens;
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

std::string CharacterTokenizer::decode(const std::vector<int64_t>& tokens) const {
    std::string out;
    out.reserve(tokens.size());
    for (auto token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= itos_.size()) {
            throw std::runtime_error("Token out of range");
        }
        out.push_back(itos_[token]);
    }
    return out;
}

TextDataset::TextDataset(torch::Tensor tokens, std::size_t seq_len)
    : tokens_(std::move(tokens)), seq_len_(seq_len) {}

torch::data::Example<> TextDataset::get(std::size_t index) {
    auto x = tokens_.narrow(0, index, seq_len_);
    auto y = tokens_.narrow(0, index + 1, seq_len_);
    return {x.clone(), y.clone()};
}

torch::optional<std::size_t> TextDataset::size() const {
    return tokens_.size(0) - seq_len_;
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

