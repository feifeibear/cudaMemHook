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
// IPC的memory handler通过 http通信
void sendSharedCache(void * shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt) {

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &s_mem->memHandle, (void *) shared_ptr));
    std::cerr << "after cudaIpcGetMemHandle" << std::endl;
    procBarrier(barrier, cnt);
    cudaEvent_t event;
    checkCudaErrors(cudaIpcOpenEventHandle(&event, s_mem->eventHandle));

    // b.2: wait until all kernels launched and events recorded
    procBarrier(barrier, cnt);
    checkCudaErrors(cudaEventSynchronize(event));
    // b.3
    procBarrier(barrier, cnt);
    //检查改写的显存
}

// shared_ptr接收其他进程数据的显存地址
// s_mem存储在共享内存空间的环境变量
// IPC的memory handler通过 http通信
void recvSharedCache(void * shared_ptr, ipcCUDA_t* s_mem, ipcBarrier_t * barrier, int cnt) {

    checkCudaErrors(cudaSetDevice(0));
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess));
    checkCudaErrors(cudaIpcGetEventHandle((cudaIpcEventHandle_t *) &s_mem->eventHandle, event));

    procBarrier(barrier, cnt);
    std::cerr << "before cudaIpcOpenMemHandle" << std::endl;
    checkCudaErrors(cudaIpcOpenMemHandle((void **) &shared_ptr, s_mem->memHandle,
                                             cudaIpcMemLazyEnablePeerAccess));
    //改写显存
    checkCudaErrors(cudaEventRecord(event));

    // b.2
    procBarrier(barrier, cnt);

    checkCudaErrors(cudaIpcCloseMemHandle(shared_ptr));

    // b.3: wait till all the events are used up by proc g_processCount - 1
    procBarrier(barrier, cnt);

    checkCudaErrors(cudaEventDestroy(event));
}

// void sendSharedCache(void * shared_ptr) {
//     std::lock_guard<std::recursive_mutex> lock(mutex);
//     cudaIpcMemHandle_t shared_cache_handle;

//     // Pack CUDA pointer
//     cudaError_t err = cudaIpcGetMemHandle(&shared_cache_handle, shared_ptr);
//     if (err != cudaSuccess) {
//         perror("pack_shared_cache");
//         exit(EXIT_FAILURE);
//     }

//     // Accept connection
//     int server_fd, conn_fd, valread;
//     int opt = 1;
//     struct sockaddr_in address;
//     int addrlen = sizeof(address);
//     if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
//         perror("socket failed");
//         exit(EXIT_FAILURE);
//     }
//     if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
//         perror("setsockopt");
//         exit(EXIT_FAILURE);
//     }
//     address.sin_family = AF_INET;
//     address.sin_addr.s_addr = INADDR_ANY;
//     address.sin_port = htons( PORT );
//     if (bind(server_fd, (struct sockaddr *)&address,  sizeof(address)) < 0) {
//         perror("bind failed");
//         exit(EXIT_FAILURE);
//     }
//     if (listen(server_fd, 1) < 0) {
//         perror("listen");
//         exit(EXIT_FAILURE);
//     }
//     if ((conn_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
//         perror("accept");
//         exit(EXIT_FAILURE);
//     }

//     // Send the packed pointer
//     write(conn_fd, (void*)(&shared_cache_handle), sizeof(cudaIpcMemHandle_t));

//     close(conn_fd);
//     close(server_fd);
// }
// void recvSharedCache(void* shared_ptr) {
//     std::lock_guard<std::recursive_mutex> lock(mutex);
//     cudaIpcMemHandle_t shared_cache_handle;

//     // Connect
//     int conn_fd = 0;
//     struct sockaddr_in serv_addr;
//     if ((conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
//         printf("\n Socket creation error \n");
//         exit(EXIT_FAILURE);
//     }
//     serv_addr.sin_family = AF_INET;
//     serv_addr.sin_port = htons(PORT);
//     if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
//         printf("\nInvalid address/ Address not supported \n");
//         exit(EXIT_FAILURE);
//     }
//     if (connect(conn_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
//         printf("\nConnection Failed \n");
//         exit(EXIT_FAILURE);
//     }

//     // Receive packed pointer
//     read(conn_fd, (void*)(&shared_cache_handle), sizeof(cudaIpcMemHandle_t));

//     // Extract the pointer
//     cudaError_t err = cudaIpcOpenMemHandle(&shared_ptr, shared_cache_handle, cudaIpcMemLazyEnablePeerAccess);
//     if (err != cudaSuccess) {
//         perror("extract_shared_cache");
//         exit(EXIT_FAILURE);
//     }

//     close(conn_fd);
// }


} // namespace ipc
} // namespace wxgpumemmgr
