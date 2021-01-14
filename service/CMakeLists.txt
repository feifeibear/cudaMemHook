find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(cuda_alloc_client SHARED cuda_alloc_client.cpp messages.cpp)
target_include_directories(cuda_alloc_client PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuda_alloc_client PUBLIC rpc ${CUDART_LIBRARY} memory)

add_executable(server_main cuda_alloc_server.cpp messages.cpp)
target_include_directories(server_main PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(server_main PUBLIC th_loguru rpc ${CUDART_LIBRARY} memory)


add_executable(client_test client_test.cpp)
target_link_libraries(client_test PUBLIC ${CUDART_LIBRARY} cuda_alloc_client)
