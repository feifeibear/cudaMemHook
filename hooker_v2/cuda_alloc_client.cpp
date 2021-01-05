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

CudaAllocClient CudaAllocClient::CreateClient() {
  const std::string server = "localhost:50051"; // TODO: only for test
  auto channel =
      grpc::CreateChannel(server, grpc::InsecureChannelCredentials());
  CudaAllocClient client;
  client.stub_ = CudaAllocator::NewStub(channel);
  return client;
}

uintptr_t CudaAllocClient::Malloc(size_t size) {
  LOG_S(INFO) << "[CudaAllocClient::Malloc], invoke with size = " << size;
  MallocRequest request;
  request.set_size(size);
  MallocReply reply;
  grpc::ClientContext context;
  auto status = stub_->Malloc(&context, request, &reply);
  assert(status.ok());
  void *ptr;
  cudaIpcMemHandle_t mem_handle;
  memcpy(&mem_handle, reply.mem_handle().data(), sizeof(mem_handle));
  cudaIpcOpenMemHandle(&ptr, mem_handle, cudaIpcMemLazyEnablePeerAccess);
  LOG_S(INFO) << "[CudaAllocClient::Malloc] get ptr = " << ptr;
  return reinterpret_cast<uintptr_t>(ptr);
}

void CudaAllocClient::Free(uintptr_t ptr) {
  LOG_S(INFO) << "[CudaAllocClient::Free], invoke with ptr = " << ptr;
  cudaIpcCloseMemHandle(reinterpret_cast<void *>(ptr));
  FreeRequest request;
  request.set_ptr_to_free(ptr);
  FreeReply reply;
  grpc::ClientContext context;
  auto status = stub_->Free(&context, request, &reply);
  assert(status.ok());
}

extern "C" {

int Malloc(uintptr_t *ptr, size_t size) {
  static CudaAllocClient client = CudaAllocClient::CreateClient();
  *ptr = client.Malloc(size);
  return 0; // TODO
}

int Free(uintptr_t ptr) {
  static CudaAllocClient client = CudaAllocClient::CreateClient();
  client.Free(ptr);
  return 0; // TODO;
}

void *Dlsym(void *handle, const char *symbol) {
  if (strcmp(symbol, "cudaMalloc") == 0) {
    return reinterpret_cast<void *>(Malloc);
  } else if (strcmp(symbol, "cudaFree") == 0) {
    return reinterpret_cast<void *>(Free);
  }
  return nullptr;
}
}
} // namespace service
} // namespace turbo_hooker
