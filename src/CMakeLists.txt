cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fsanitize=address,leak,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Enable march=native if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
if (HAS_MARCH_NATIVE)
  target_compile_options(${PROJECT_NAME} INTERFACE -march=native)
endif()

# ==== Non compatible compile flags ====
check_cxx_compiler_flag(-static-libsan HAS_LIBSAN)
if (HAS_LIBSAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libsan")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -static-libasan")
endif()

# ==== Test target ====
add_executable(ht_tests tests.cpp)

include(${PROJECT_SOURCE_DIR}/thirdparty/googletest.cmake)
target_link_libraries(ht_tests ${PROJECT_NAME} ${GOOGLETEST_LIBRARY})

# enable ctest support (i.e., test discovery)
include(GoogleTest)
gtest_discover_tests(ht_tests)

# ==== Benchmarks target ====
add_executable(ht_benchmarks benchmarks.cpp)

include(${PROJECT_SOURCE_DIR}/thirdparty/hashing.cmake)
include(${PROJECT_SOURCE_DIR}/thirdparty/learned_hashing.cmake)
include(${PROJECT_SOURCE_DIR}/thirdparty/exotic_hashing.cmake)

include(${PROJECT_SOURCE_DIR}/thirdparty/googlebenchmark.cmake)
target_link_libraries(ht_benchmarks PRIVATE ${PROJECT_NAME} ${GOOGLEBENCHMARK_LIBRARY} ${HASHING_LIBRARY} ${LEARNED_HASHING_LIBRARY} ${EXOTIC_HASHING_LIBRARY})

