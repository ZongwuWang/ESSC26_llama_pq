set(TARGET pq-selftest)
add_executable(${TARGET} pq-selftest.cpp)
target_link_libraries(${TARGET} PRIVATE ggml-cpu ggml)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
