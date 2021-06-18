cmake_minimum_required(VERSION 3.19)
project("hashtable" VERSION 1.0
        DESCRIPTION "A header only c++ library that exposes various hashtable implementations meant to benchmark the real world performance of different hashing schemas"
        HOMEPAGE_URL "https://github.com/andreaskipf/hashing")

# Declare library & directories to include.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_compile_options(${PROJECT_NAME} INTERFACE cxx_std_20)

# Specify how to install hashing
install(FILES hashtable.hpp TYPE INCLUDE)
install(DIRECTORY include TYPE INCLUDE)
