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

#include "gpuipc.h"
#include "catch2/catch.hpp"
#include "cuda_runtime.h"
#include <sys/types.h>
#include <unistd.h>

namespace wxgpumemmgr {
namespace ipc {

TEST_CASE("cuda ipc", "init") {
    pid_t pid;

    pid = fork();
    constexpr size_t N = 5;
    if (pid == 0) {
        printf("Child process!\n");
        std::unique_ptr<float[]> host_send_data(new float [N]);
        for(size_t i = 0; i < N; ++i) {
            host_send_data[i] = 1. * i;
        }
        void* dev_send_data;
        cudaMalloc(&dev_send_data, N*sizeof(float));
        cudaMemcpy(dev_send_data, host_send_data.get(), N*sizeof(float), cudaMemcpyHostToDevice);
        sendSharedCache(dev_send_data);
        sleep(15);
    } else if (pid > 0) {
        printf("Parent process!\n");
        void* dev_recv_data;
        sleep(5);
        recvSharedCache(dev_recv_data);
        std::unique_ptr<float[]> host_recv_data(new float [N]);
        cudaMemcpy(host_recv_data.get(), dev_recv_data, N*sizeof(float), cudaMemcpyDeviceToHost);
        for(size_t i = 0; i < N; ++i) {
            printf("%f\n", host_recv_data[i]);
        }
        sleep(5);
    } else {
        printf("Error!n");
    }
}


}  // namespace ipc
}  // namespace wxgpumemmgr
