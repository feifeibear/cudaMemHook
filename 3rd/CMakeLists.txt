add_library(loguru SHARED loguru/loguru.cpp)
target_include_directories(loguru PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/loguru)
target_compile_definitions(loguru PUBLIC -DLOGURU_WITH_STREAMS=1)
target_link_libraries(loguru PUBLIC ${CMAKE_DL_LIBS})

add_subdirectory(abseil-cpp)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(rpclib)
add_subdirectory(Catch2)
