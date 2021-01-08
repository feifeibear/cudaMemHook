find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(cuda_alloc_client SHARED cuda_alloc_client.cpp messages.cpp)
target_include_directories(cuda_alloc_client PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_alloc_client PUBLIC loguru rpc ${CUDART_LIBRARY} memory)

add_executable(server_main cuda_alloc_server.cpp messages.cpp)
target_include_directories(server_main PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(server_main PUBLIC loguru rpc ${CUDART_LIBRARY} memory)
