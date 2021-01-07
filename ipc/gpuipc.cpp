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

//
// Created by Jiarui Fang on 2020/12/15.
//

#include <mutex>
#include "gpuipc.h"
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <memory>
#include "helper_cuda.h"

namespace wxgpumemmgr {
namespace ipc {
static std::recursive_mutex mutex;

bool g_procSense = false;

void procBarrier(ipcBarrier_t *g_barrier, int g_processCount)
{
    int newCount = __sync_add_and_fetch(&g_barrier->count, 1);

    if (newCount == g_processCount)
    {
        g_barrier->count = 0;
        g_barrier->sense = !g_procSense;
    }
    else
    {
        while (g_barrier->sense == g_procSense)
        {
            if (!g_barrier->allExit)
            {
                sched_yield();
            }
            else
            {
                exit(EXIT_FAILURE);
            }
        }
    }

    g_procSense = !g_procSense;
}

// void* Server::getMemoryCache(size_t offset) {
//     //找到内存碎片
//     return static_cast<void*>(static_cast<char*>(memory_cache_) + offset);
// }
// Server::~Server() {
//     checkCudaErrors(cudaFree(memory_cache_));
// }
// void Server::initSharedCache() {
//     std::unique_ptr<int[]> h_memory = std::make_unique<int[]>(capacity_ / sizeof(int));
//     for (auto i = 0; i < capacity_ / sizeof(int); ++i) {
//         h_memory[i] = i;
//     }
//     checkCudaErrors(cudaMalloc((void **) &memory_cache_, capacity_));
//     checkCudaErrors(cudaMemcpy((void *) memory_cache_,
//                                 (void *) h_memory.get(),
//                                 capacity_,
//                                 cudaMemcpyHostToDevice));
// }


// shared_ptr发送给其他进程的显存地址
// s_mem存储在共享内存空间的环境变量
void sendSharedCache(void * shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt) {

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &s_mem->memHandle, (void *) shared_ptr));

    // b.1 prepare memory handle finished
    procBarrier(barrier, cnt);
    cudaEvent_t event;

    // b.2 wait event handle
    procBarrier(barrier, cnt);
    checkCudaErrors(cudaIpcOpenEventHandle(&event, s_mem->eventHandle));

    // b.3: wait until all kernels launched and events recorded
    procBarrier(barrier, cnt);
    checkCudaErrors(cudaEventSynchronize(event));
    // b.4 close event on server
    procBarrier(barrier, cnt);
}

// shared_ptr接收其他进程数据的显存地址
// s_mem存储在共享内存空间的环境变量
void recvSharedCache(void ** shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt) {

    checkCudaErrors(cudaSetDevice(0));
    // b.1 wait memory handle finished
    procBarrier(barrier, cnt);
    checkCudaErrors(cudaIpcOpenMemHandle((void **) shared_ptr, s_mem->memHandle,
                                             cudaIpcMemLazyEnablePeerAccess));
}

void cleanSharedCache(void ** shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt) {
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess));
    checkCudaErrors(cudaIpcGetEventHandle((cudaIpcEventHandle_t *) &s_mem->eventHandle, event));

    // b.2 prepare event handle
    procBarrier(barrier, cnt);

    // TODO(jiaruifang) 对显存进行清除，并告诉server shared_ptr不再用
    checkCudaErrors(cudaEventRecord(event));

    // b.3 all kernels launched event recorded
    procBarrier(barrier, cnt);

    checkCudaErrors(cudaIpcCloseMemHandle(*shared_ptr));

    // b.4: wait till all the events are used up by server
    procBarrier(barrier, cnt);

    checkCudaErrors(cudaEventDestroy(event));
}


} // namespace ipc
} // namespace wxgpumemmgr
