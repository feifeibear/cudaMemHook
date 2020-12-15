cmake_minimum_required(VERSION 3.12)
project(gpuMemMgr)

set(CMAKE_CXX_STANDARD 14)

add_library(cuda_hooker SHARED
        dlsymhooker.cpp
        realdlsym.cpp
        cudahooker.cpp
        cudakernels.cpp)

target_compile_definitions(cuda_hooker
        PRIVATE
        _GNU_SOURCE=1
        )

target_link_libraries(cuda_hooker
        PRIVATE
        ${CMAKE_DL_LIBS}
        )

install(TARGETS cuda_hooker
        LIBRARY DESTINATION lib
        )
