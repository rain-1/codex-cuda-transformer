#include "config.hpp"

#include <stdexcept>

ModelConfig preset_from_name(const std::string& name) {
    if (name == "pico") {
        return ModelConfig{512, 256, 256, 6, 8, 1024, 0.1};
    } else if (name == "nano") {
        return ModelConfig{2048, 512, 512, 12, 8, 2048, 0.1};
    } else if (name == "micro") {
        return ModelConfig{4096, 1024, 1024, 24, 16, 8192, 0.1};
    }
    throw std::runtime_error("Unknown model preset: " + name);
}

