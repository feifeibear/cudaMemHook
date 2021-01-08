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
#include "messages.h"
#include "rpc/client.h"
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>

namespace turbo_hook {
namespace service {

static CudaAllocClient gClient = []() {
  return CudaAllocClient("localhost", 50051);
}();

struct CudaAllocClient::Impl {
  Impl(const std::string &addr, uint16_t port) : client_(addr, port) {}

  int Malloc(uintptr_t *ptr, size_t size) {
    auto reply = client_.call("Malloc", MallocRequest{size}).as<MallocReply>();
    cudaIpcMemHandle_t handle;
    memcpy(&handle, reply.ipc_handle_bytes_.data(), sizeof(handle));
    void *ipc_mem;
    assert(cudaIpcOpenMemHandle(&ipc_mem, handle,
                                cudaIpcMemLazyEnablePeerAccess) == 0);
    *ptr = reinterpret_cast<uintptr_t>(ipc_mem) + reply.offset_;
    std::lock_guard<std::mutex> lck(mtx_);
    free_req_[*ptr] = FreeRequest{reply.original_ptr_, reply.offset_};
    LOG_S(INFO) << "[Client::Malloc] return with ptr=" << *ptr;
    return 0;
  }

  int Free(uintptr_t ptr) {
    LOG_S(INFO) << "[Client::Malloc] invoked with ptr=" << ptr;
    if (ptr == 0) {
      return 0;
    }
    std::lock_guard<std::mutex> lck(mtx_);
    const FreeRequest &req = free_req_.at(ptr);
    void *ipc_mem = reinterpret_cast<void *>(ptr - req.offset_);
    assert(cudaIpcCloseMemHandle(ipc_mem) == 0);
    client_.call("Free", req);
    free_req_.erase(ptr);
    return 0;
  }

private:
  rpc::client client_;
  std::unordered_map<uintptr_t, FreeRequest> free_req_;
  std::mutex mtx_;
};

CudaAllocClient::CudaAllocClient(const std::string &addr, uint16_t port)
    : m_(new Impl(addr, port)) {}

int CudaAllocClient::Malloc(uintptr_t *ptr, size_t size) {
  return m_->Malloc(ptr, size);
}

int CudaAllocClient::Free(uintptr_t ptr) { return m_->Free(ptr); }

extern "C" {

int Malloc(uintptr_t *ptr, size_t size) { return gClient.Malloc(ptr, size); }

int Free(uintptr_t ptr) { return gClient.Free(ptr); }

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
} // namespace turbo_hook
