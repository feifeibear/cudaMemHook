cmake_minimum_required(VERSION 3.12)
project(turbo_hooker LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(3rd)
add_subdirectory(core)
add_subdirectory(hooker)
add_subdirectory(ipc)
