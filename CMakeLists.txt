cmake_minimum_required(VERSION 3.12)
project(turbo_hooker)
set(CMAKE_CXX_STANDARD 14)

enable_language(CUDA)

add_subdirectory(3rd)
#add_subdirectory(hooker)
add_subdirectory(hooker_v2)
