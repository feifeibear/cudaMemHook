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
        loguru
        ${CMAKE_DL_LIBS}
        )

install(TARGETS cuda_hooker
        LIBRARY DESTINATION lib
        )
