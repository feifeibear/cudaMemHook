add_library(cuda_hooker_v2 SHARED dlsym_hooker.cpp real_dlsym.cpp)

target_link_libraries(cuda_hooker_v2 PUBLIC ${CMAKE_DL_LIBS})
find_package(Threads REQUIRED)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(server_main cuda_alloc_server.cpp alloc.pb.cc alloc.grpc.pb.cc)

target_include_directories(server_main PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(server_main PUBLIC cuda_hooker_v2 gRPC::grpc++_reflection gRPC::grpc++ protobuf::libprotobuf ${CUDART_LIBRARY})
