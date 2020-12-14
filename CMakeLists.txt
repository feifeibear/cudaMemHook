cmake_minimum_required(VERSION 3.12)
project(gpuMemMgr)

set(CMAKE_CXX_STANDARD 14)

add_library(cuda_hooker SHARED dlsym_hook.cpp realdlsym.cpp)
target_link_libraries(cuda_hooker ${CMAKE_DL_LIBS})
add_compile_options(cuda_hooker -fPIC)
