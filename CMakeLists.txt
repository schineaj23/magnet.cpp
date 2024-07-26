cmake_minimum_required(VERSION 3.12)
project("magnet" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)

set(MAGNET_LIB magnet)

add_subdirectory(ggml)

# add_library(${MAGNET_LIB} STATIC magnet.cpp magnet.h)
add_executable(${MAGNET_LIB} magnet.cpp magnet.h)

target_link_libraries(${MAGNET_LIB} PUBLIC ggml)
target_include_directories(${MAGNET_LIB} PUBLIC .)
target_compile_features(${MAGNET_LIB} PUBLIC cxx_std_11)

if(GGML_VULKAN)
    add_compile_definitions(GGML_USE_VULKAN)
endif()

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()

# Copy the ggml DLL to the output directory (if on windows)
add_custom_command(TARGET ${MAGNET_LIB} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:ggml>"
        $<TARGET_FILE_DIR:${MAGNET_LIB}>)