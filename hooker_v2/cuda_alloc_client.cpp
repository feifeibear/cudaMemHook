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

#include "cuda_alloc_client.h"
#include "loguru.hpp"
#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

namespace turbo_hooker {
namespace service {

static CudaAllocClient gClient = []() {
  const std::string server = "localhost:50051"; // TODO: only for test
  return CudaAllocClient(server);
}();

CudaAllocClient::CudaAllocClient(const std::string &server_address) {
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  stub_ = CudaAllocator::NewStub(channel);
}

uintptr_t CudaAllocClient::Malloc(size_t size) {
  LOG_S(INFO) << "[CudaAllocClient::Malloc], invoke with size = " << size;
  MallocRequest request;
  request.set_size(size);
  MallocReply reply;
  grpc::ClientContext context;
  auto status = stub_->Malloc(&context, request, &reply);
  assert(status.ok());
  cudaIpcMemHandle_t mem_handle;
  memcpy(&mem_handle, reply.mem_handle().data(), sizeof(mem_handle));
  void *ptr;
  assert(cudaIpcOpenMemHandle(&ptr, mem_handle,
                              cudaIpcMemLazyEnablePeerAccess) == 0);
  LOG_S(INFO) << "[CudaAllocClient::Malloc] get ptr = " << ptr;
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  allocations_[ptr_int] = reply.allocation();
  return ptr_int;
}

void CudaAllocClient::Free(uintptr_t ptr) {
  LOG_S(INFO) << "[CudaAllocClient::Free], invoke with ptr = " << ptr;
  assert(cudaIpcCloseMemHandle(reinterpret_cast<void *>(ptr)) == 0);
  FreeRequest request;
  request.set_ptr_to_free(allocations_.at(ptr).ptr());
  FreeReply reply;
  grpc::ClientContext context;
  auto status = stub_->Free(&context, request, &reply);
  assert(status.ok());
}

extern "C" {

int Malloc(uintptr_t *ptr, size_t size) {
  *ptr = gClient.Malloc(size);
  return 0; // TODO
}

int Free(uintptr_t ptr) {
  gClient.Free(ptr);
  return 0; // TODO;
}

void *Dlsym(void *handle, const char *symbol) {
  if (strcmp(symbol, "cuMemAlloc_v2") == 0) {
    return reinterpret_cast<void *>(Malloc);
  } else if (strcmp(symbol, "cuMemFree_v2") == 0) {
    return reinterpret_cast<void *>(Free);
  }
  return nullptr;
}
}
} // namespace service
} // namespace turbo_hooker
