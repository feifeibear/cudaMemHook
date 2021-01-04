add_library(cuda_hooker_v2 SHARED dlsym_hooker.cpp real_dlsym.cpp)

find_package(gRPC CONFIG REQUIRED)

add_executable(server_main cuda_alloc_server.cpp)

target_link_libraries(server_main gRPC::grpc++_reflection gRPC::grpc++ protobuf::libprotobuf)