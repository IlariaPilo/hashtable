cmake_minimum_required(VERSION 3.19)
project("hashtable" VERSION 1.0
        DESCRIPTION "A header only c++ library that exposes various hashtable implementations meant to benchmark the real world performance of different hashing schemas"
        HOMEPAGE_URL "https://github.com/DominikHorn/hashtable")

# Declare library & directories to include.
add_library(${PROJECT_NAME} INTERFACE)
target_sources(${PROJECT_NAME} INTERFACE hashtable.hpp include/)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Benchmark & Test code
get_directory_property(hasParent PARENT_DIRECTORY)
if (NOT hasParent)
  add_subdirectory(src)
endif()
