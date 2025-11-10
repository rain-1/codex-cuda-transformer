cmake_minimum_required(VERSION 3.18)
project(codex_cuda_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(transformer_train
    src/main.cpp
    src/model.cpp
    src/dataset.cpp
    src/config.cpp)

target_include_directories(transformer_train PRIVATE include)

target_link_libraries(transformer_train PRIVATE ${TORCH_LIBRARIES})

set_property(TARGET transformer_train PROPERTY CXX_STANDARD 17)

