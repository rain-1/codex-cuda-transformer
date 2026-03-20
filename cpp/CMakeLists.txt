cmake_minimum_required(VERSION 3.18)
project(codex_cuda_transformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(transformer_train
    src/main.cpp
    src/model.cpp
    src/dataset.cpp
    src/config.cpp
    src/ops.cu)

target_include_directories(transformer_train PRIVATE include)

target_link_libraries(transformer_train PRIVATE CUDA::cudart)

set_target_properties(transformer_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

