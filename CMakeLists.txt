cmake_minimum_required(VERSION 3.14)
project(essc_llama_pq_ae LANGUAGES CXX)

set(LLAMA_SRC "${CMAKE_CURRENT_LIST_DIR}/llama.cpp" CACHE PATH "llama.cpp source directory")
set(LLAMA_BUILD "${CMAKE_CURRENT_LIST_DIR}/build" CACHE PATH "llama.cpp build directory")

add_executable(llama_pq llama_pq.cpp)
target_compile_features(llama_pq PRIVATE cxx_std_17)
target_compile_options(llama_pq PRIVATE -Wall -Wextra -pedantic)
target_include_directories(llama_pq PRIVATE
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/ggml/include"
)
target_link_directories(llama_pq PRIVATE "${LLAMA_BUILD}/bin")
target_link_libraries(llama_pq PRIVATE llama ggml ggml-cpu ggml-base pthread dl m)
set_target_properties(llama_pq PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD}/bin"
    INSTALL_RPATH "${LLAMA_BUILD}/bin"
)
