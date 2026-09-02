set(TARGET llama-pq-chat)
add_executable(${TARGET} pq-chat.cpp)
target_link_libraries(${TARGET} PRIVATE llama Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
