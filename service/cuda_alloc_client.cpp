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
#include "messages.h"
#include "rpc/client.h"
#include <cuda_runtime.h>
#include <iostream>
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
    std::cerr << "[Client::Malloc] return with ptr=" << *ptr << std::endl;
    return 0;
  }

  int Free(uintptr_t ptr) {
    std::cerr << "[Client::Malloc] invoked with ptr=" << ptr << std::endl;
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

  // Register the client to the server. Get the CUDA IPC memory handle.
  int Register(pid_t pid) {
    auto reply = client_.call("Register", RegistRequest{pid}).as<RegistReply>();
    cudaIpcMemHandle_t handle;
    memcpy(&handle, reply.ipc_handle_bytes_.data(), sizeof(handle));
    assert(cudaIpcOpenMemHandle((void **)&ipc_memory_, handle,
                                cudaIpcMemLazyEnablePeerAccess) == 0);
    std::cerr << "[Client::Register] Success Register Pid " << pid;
    return 0;
  }

  // send the request to the server to find a memory gap from the global memory
  // pool.
  uintptr_t uMalloc(pid_t pid, size_t size) {
    auto reply =
        client_.call("uMalloc", uMallocRequest{pid, size}).as<uMallocReply>();
    auto offset = reply.offset_;
    if (offset != -1U) {
      uintptr_t ret = ipc_memory_ + offset;
      allocation_records_[ret] = offset;
      return ret;
    } else {
      return -1U;
    }
  }

  // free the ownership of pid on addr from the global memory pool.
  void uFree(pid_t pid, uintptr_t addr) {
    auto it = allocation_records_.find(addr);
    if (it != allocation_records_.end()) {
      client_.call("uFree", uMallocRequest{pid, it->second});
      allocation_records_.erase(it);
    } else {
      std::cerr << "uFree an invalid memory addr" << std::endl;
    }
  }

private:
  rpc::client client_;
  std::unordered_map<uintptr_t, FreeRequest> free_req_;
  std::mutex mtx_;

  std::unordered_map<uintptr_t, size_t> allocation_records_;

  uintptr_t ipc_memory_;
};

CudaAllocClient::CudaAllocClient(const std::string &addr, uint16_t port)
    : m_(new Impl(addr, port)) {}

int CudaAllocClient::Malloc(uintptr_t *ptr, size_t size) {
  return m_->Malloc(ptr, size);
}

int CudaAllocClient::Free(uintptr_t ptr) { return m_->Free(ptr); }

int CudaAllocClient::Register(pid_t pid) { return m_->Register(pid); }

uintptr_t CudaAllocClient::uMalloc(pid_t pid, size_t size) {
  return m_->uMalloc(pid, size);
}

void CudaAllocClient::uFree(pid_t pid, uintptr_t addr) {
  return m_->uFree(pid, addr);
}

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
int Register(pid_t pid) { return gClient.Register(pid); }

uintptr_t uMalloc(pid_t pid, size_t size) { return gClient.uMalloc(pid, size); }

void uFree(pid_t pid, uintptr_t addr) { return gClient.uFree(pid, addr); };
}

} // namespace service
} // namespace turbo_hook
