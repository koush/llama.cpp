message(STATUS "Using Tensor Parallel backend")

ggml_add_backend_library(ggml-tp
                         ggml-tp.cpp
                        )

if (WIN32)
    target_link_libraries(ggml-tp PRIVATE ws2_32)
endif()
