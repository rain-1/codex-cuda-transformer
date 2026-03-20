#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class CharacterTokenizer {
  public:
    explicit CharacterTokenizer(const std::string& text);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::size_t vocab_size() const { return itos_.size(); }

  private:
    std::vector<char> itos_;
    std::unordered_map<char, int> stoi_;
};

struct Batch {
    std::vector<int> input;
    std::vector<int> target;
    int batch_size = 0;
    int seq_len = 0;
};

class TextDataset {
  public:
    TextDataset(std::vector<int> tokens, std::size_t seq_len);

    Batch sample(std::mt19937& rng, int batch_size) const;
    Batch sample_sequential(std::size_t offset, int batch_size) const;
    std::size_t size() const { return tokens_.size(); }
    std::size_t seq_len() const { return seq_len_; }

  private:
    std::vector<int> tokens_;
    std::size_t seq_len_;
};

std::string read_file(const std::string& path);

