cmake_minimum_required(VERSION 3.12)
project(turbo_hooker LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(3rd)
add_subdirectory(core)
add_subdirectory(memory)
add_subdirectory(hook_v2)
add_subdirectory(service)
# add_subdirectory(ipc)
#add_subdirectory(hooker)
