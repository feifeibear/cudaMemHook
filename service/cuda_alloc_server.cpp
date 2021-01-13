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
#include "loguru.hpp"
#include "memory/naive_scheduler.hpp"
#include "messages.h"
#include "rpc/server.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace turbo_hook {
namespace service {

struct Allocation {
  cudaIpcMemHandle_t ipc_handle_;
  uintptr_t original_ptr_;
  size_t offset_;
};

class Allocator {
public:
  Allocation Malloc(size_t size) {
    void *ptr;
    assert(cudaMalloc(&ptr, size) == 0);
    cudaIpcMemHandle_t handle;
    assert(cudaIpcGetMemHandle(&handle, ptr) == 0);
    return Allocation{handle, reinterpret_cast<uintptr_t>(ptr), 0};
  }

  void Free(uintptr_t original_ptr, size_t offset) {
    assert(cudaFree(reinterpret_cast<void *>(original_ptr)) == 0);
  }
};

struct CudaAllocServer::Impl {
  explicit Impl(uint16_t port, size_t capcaity = 1024 * 1024 * 1024)
      : server_(port),
        memory_scheduler_(new turbo_hook::memory::NaiveScheduler(capcaity)) {

    memory_pool_ = allocator_.Malloc(capcaity);

    server_.bind("Malloc", [&](const MallocRequest &req) -> MallocReply {
      auto allocation = allocator_.Malloc(req.size_);
      std::ostringstream oss;
      oss.write(reinterpret_cast<char *>(&allocation.ipc_handle_),
                sizeof(allocation.ipc_handle_));
      MallocReply reply{allocation.original_ptr_, oss.str(),
                        allocation.offset_};
      LOG_S(INFO) << "[Server::Malloc] Return with ptr=" << reply.original_ptr_
                  << " offset=" << reply.offset_;
      return reply;
    });

    server_.bind("Free", [&](const FreeRequest &req) -> void {
      LOG_S(INFO) << "[Server::Free] Invoked with ptr=" << req.original_ptr_
                  << " offset=" << req.offset_;
      allocator_.Free(req.original_ptr_, req.offset_);
    });

    // client向server注册，获得gpu内存分配句柄
    server_.bind("Register", [&](const RegistRequest &req) -> RegistReply {
      auto pid = req.pid_;
      std::ostringstream oss;
      oss.write(reinterpret_cast<char *>(&memory_pool_.ipc_handle_),
                sizeof(memory_pool_.ipc_handle_));
      RegistReply reply{memory_pool_.original_ptr_, oss.str()};
      LOG_S(INFO) << "[Server::Register] regist pid  " << pid
                  << " original addr=" << memory_pool_.original_ptr_;
      return reply;
    });
  }

  void Run() { server_.run(); }

private:
  rpc::server server_;
  Allocator allocator_;
  std::unique_ptr<turbo_hook::memory::IScheduler> memory_scheduler_;
  Allocation memory_pool_;
};

CudaAllocServer::CudaAllocServer(uint16_t port) : m_(new Impl(port)) {}

void CudaAllocServer::Run() { m_->Run(); }

} // namespace service
} // namespace turbo_hook

int main(int argc, char **argv) {
  turbo_hook::service::CudaAllocServer server(50051);
  LOG_S(INFO) << "Server start.";
  server.Run();
  return 0;
}
