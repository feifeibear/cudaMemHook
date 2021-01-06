// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "cuda_alloc_server.h"
#include "alloc.pb.h"
#include "loguru.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

// using DlsymFn = void *(void *, const char *);
//
// extern "C" {
//// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
//// function
// void *__libc_dlsym(void *map, const char *name);
//}
//
// namespace turbo_hooker {
// template <typename FnPtrT> constexpr auto func_cast(void *ptr) noexcept {
//  return reinterpret_cast<FnPtrT>(reinterpret_cast<intptr_t>(ptr));
//}
//
// DlsymFn *GetRealDlsym() noexcept {
//  return func_cast<DlsymFn *>(
//      __libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym"));
//}
//}

namespace turbo_hooker {
namespace service {

// CudaAllocServer::CudaAllocServer() {
//  const std::string cuda_lib = "libcuda.so";
//  auto handle = dlopen(cuda_lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
//  auto real_dlsym = GetRealDlsym();
//  cuda_malloc_ =
//     reinterpret_cast<CudaMallocFn *>(real_dlsym(handle, "cuMemAlloc_v2"));
//  cuda_free_ =
//      reinterpret_cast<CudaFreeFn *>(real_dlsym(handle, "cuMemFree_v2"));
//}

CudaAllocServer::CudaAllocServer() = default;

grpc::Status CudaAllocServer::Malloc(grpc::ServerContext *context,
                                     const MallocRequest *request,
                                     MallocReply *response) {
  LOG_S(INFO) << "[CudaAllocServer::Malloc] << invoke with size = "
              << request->size();
  void *ptr;
  cudaIpcMemHandle_t mem_handle;
  size_t s = request->size();
  assert(cudaMalloc(&ptr, s) == 0);
  assert(cudaIpcGetMemHandle(&mem_handle, ptr) == 0);
  response->set_mem_handle(&mem_handle, sizeof(mem_handle));
  auto *alloc = response->mutable_allocation();
  alloc->set_ptr(reinterpret_cast<uintptr_t>(ptr));
  LOG_S(INFO) << "[CudaAllocServer::Malloc] << return with new malloced ptr = "
              << ptr;
  return grpc::Status::OK;
}

grpc::Status CudaAllocServer::Free(grpc::ServerContext *context,
                                   const FreeRequest *request,
                                   FreeReply *response) {
  uintptr_t to_free = static_cast<intptr_t>(request->ptr_to_free());
  LOG_S(INFO) << "[CudaAllocServer::Free] << invoke with ptr = " << to_free;
  assert(cudaFree(reinterpret_cast<void *>(to_free)) == 0);
  return grpc::Status::OK;
}

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  CudaAllocServer service;

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;

  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "CudaAllocServer listening on " << server_address << std::endl;

  server->Wait();
}

} // namespace service
} // namespace turbo_hooker

int main(int argc, char **argv) {

  turbo_hooker::service::RunServer();

  return 0;
}
