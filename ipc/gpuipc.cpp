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

#include "gpuipc.h"
#include <cuda_runtime_api.h>
#include <iostream> // PipeSwitch
#include <unistd.h>  // PipeSwitch
#include <stdio.h>  // PipeSwitch
#include <sys/socket.h>  // PipeSwitch
#include <stdlib.h>  // PipeSwitch
#include <netinet/in.h>  // PipeSwitch
#include <string.h> // PipeSwitch
#include <arpa/inet.h>  // PipeSwitch
#define PORT 9001 // PipeSwitch
#define SIZE_SHARED_CACHE (12 * 1024UL * 1024UL * 1024UL) // PipeSwitch


namespace wxgpumemmgr {

static std::recursive_mutex mutex;

void sendSharedCache(void * shared_ptr) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cudaIpcMemHandle_t shared_cache_handle;

    // Pack CUDA pointer
    cudaError_t err = cudaIpcGetMemHandle(&shared_cache_handle, shared_ptr);
    if (err != cudaSuccess) {
        perror("pack_shared_cache");
        exit(EXIT_FAILURE);
    }

    // Accept connection
    int server_fd, conn_fd, valread;
    int opt = 1;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );
    if (bind(server_fd, (struct sockaddr *)&address,  sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 1) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((conn_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // Send the packed pointer
    write(conn_fd, (void*)(&shared_cache_handle), sizeof(cudaIpcMemHandle_t));

    close(conn_fd);
    close(server_fd);
}

void recvSharedCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cudaIpcMemHandle_t shared_cache_handle;

    // Connect
    int conn_fd = 0;
    struct sockaddr_in serv_addr;
    if ((conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        exit(EXIT_FAILURE);
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        exit(EXIT_FAILURE);
    }
    if (connect(conn_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        exit(EXIT_FAILURE);
    }

    // Receive packed pointer
    read(conn_fd, (void*)(&shared_cache_handle), sizeof(cudaIpcMemHandle_t));

    // Extract the pointer
    cudaError_t err = cudaIpcOpenMemHandle(&PIPESWITCH_shared_ptr, shared_cache_handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        perror("extract_shared_cache");
        exit(EXIT_FAILURE);
    }

    close(conn_fd);
}
} // namespace wxgpumemmgr
