add_library(wxipc
        gpuipc.cpp)

target_link_libraries(wxipc PUBLIC cudart cuda)

add_executable(gpuipc_test gpuipc_test.cpp)
target_link_libraries(gpuipc_test wxipc catch2_test_main)
add_test(NAME gpuipc_test COMMAND gpuipc_test)
