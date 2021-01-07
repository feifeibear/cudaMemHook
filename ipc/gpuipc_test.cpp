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

#include <sys/types.h>
#include <sys/wait.h>
#include "gpuipc.h"
#include "catch2/catch.hpp"
#include "cuda_runtime.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <cstring>

namespace wxgpumemmgr {
namespace ipc {

TEST_CASE("cuda ipc", "init") {
    constexpr int process_cnt = 2;
    ipcCUDA_t *s_mem = (ipcCUDA_t *) mmap(NULL, process_cnt * sizeof(ipcCUDA_t),
                                        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != s_mem);

    std::cerr << "I am in cuda ipc unitest" << std::endl;
    // initialize shared memory
    memset((void *) s_mem, 0, process_cnt * sizeof(*s_mem));
    int index = 0;

    ipcBarrier_t *g_barrier = (ipcBarrier_t *) mmap(NULL, sizeof(ipcBarrier_t),
                                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != g_barrier);
    memset((void *) g_barrier, 0, sizeof(*g_barrier));

    // index表示当前进程的索引
    // 父进程的s_mem[index]存储它子进程的pid
    for (int i = 1; i < process_cnt; i++)
    {
        int pid = fork();
        if (pid == 0)
        {
            //子进程
            index = i;
            break;
        } else {
            //父程序
            std::cerr << "I launch a child process " << pid << std::endl;
            s_mem[i].pid = pid;
        }
    }

    //父进程
    if (index == 0)
    {
        std::cerr << "parent process" << std::endl;
        std::unique_ptr<int[]> h_memory = std::make_unique<int[]>(100 / sizeof(int));
        for (auto i = 0; i < 100 / sizeof(int); ++i) {
            h_memory[i] = i;
        }
        void* memory_cache_;
        cudaMalloc((void **) &memory_cache_, 100);

        cudaMemcpy((void *) memory_cache_,
                                    (void *) h_memory.get(),
                                    100,
                                    cudaMemcpyHostToDevice);

        sendSharedCache(memory_cache_, &s_mem[0], g_barrier, 2);


    } else {
        std::cerr << "child process" << std::endl;
        void* cached_mem;
        recvSharedCache(&cached_mem, &s_mem[0], g_barrier, 2);

        //对显存进行操作
        //这里访问子线程的内存地址，看5个
        std::unique_ptr<int[]> h_memory = std::make_unique<int[]>(5);
        cudaMemcpy((void *) h_memory.get(),
                                (void *) cached_mem,
                                5 * sizeof(int),
                                cudaMemcpyDeviceToHost);
        for(int i = 0; i < 5; ++i) {
            std::cerr << i << " " << h_memory[i] << std::endl;
        }

        //清除内存并关闭和server连接
        cleanSharedCache(&cached_mem, &s_mem[0], g_barrier, 2);
    }

    // Cleanup and shutdown
    if (index == 0)
    {
        // wait for processes to complete
        for (int i = 1; i < process_cnt; i++)
        {
            int status;
            waitpid(s_mem[i].pid, &status, 0);
            assert(WIFEXITED(status));
        }

        printf("\nShutting down...\n");
        exit(EXIT_SUCCESS);
    }

}


}  // namespace ipc
}  // namespace wxgpumemmgr
