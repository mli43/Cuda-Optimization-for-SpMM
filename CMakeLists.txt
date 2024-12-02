cmake_minimum_required(VERSION 3.20)

project(cuspmm LANGUAGES CUDA CXX)

list(APPEND CMAKE_PREFIX_PATH "/15618/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 90)

message("cuda arch: ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES src/*.cu src/*.cpp)

add_executable(cuspmm ${SOURCES})

target_link_libraries(cuspmm "${TORCH_LIBRARIES}")

set_target_properties(cuspmm
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)