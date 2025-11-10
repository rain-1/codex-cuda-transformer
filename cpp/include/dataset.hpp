#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

class CharacterTokenizer {
  public:
    explicit CharacterTokenizer(const std::string& text);

    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& tokens) const;
    std::size_t vocab_size() const { return stoi_.size(); }

  private:
    std::vector<char> itos_;
    std::unordered_map<char, int64_t> stoi_;
};

class TextDataset : public torch::data::datasets::Dataset<TextDataset> {
  public:
    TextDataset(torch::Tensor tokens, std::size_t seq_len);

    torch::data::Example<> get(std::size_t index) override;
    torch::optional<std::size_t> size() const override;

  private:
    torch::Tensor tokens_;
    std::size_t seq_len_;
};

std::string read_file(const std::string& path);

