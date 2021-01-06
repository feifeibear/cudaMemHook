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

#pragma once
#include <stdlib.h>
// CUDA utilities and system includes
#include <cuda_runtime_api.h>

namespace wxgpumemmgr {
namespace ipc {

typedef struct ipcCUDA_st
{
    int device;
    pid_t pid;
    cudaIpcEventHandle_t eventHandle;
    cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;

typedef struct ipcBarrier_st
{
    int count;
    bool sense;
    bool allExit;
} ipcBarrier_t;

// class Server {
//   public:
//     static Server& getInstance() {
//       static Server server;
//       return server;
//     }
//     void* getMemoryCache(size_t offset);
//     ~Server();
//   private:
//     void initSharedCache();

//     Server() : memory_cache_{nullptr}, capacity_(100 * sizeof(int)) {
//       initSharedCache();
//     }
//     void* memory_cache_;
//     size_t capacity_;
// };

void sendSharedCache(void * shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt);
void recvSharedCache(void * shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt
);

} // namespace ipc
} // namespace wxgpumemmgr
