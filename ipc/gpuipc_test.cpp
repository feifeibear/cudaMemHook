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

namespace wxgpumemmgr {
namespace ipc {

TEST_CASE("cuda ipc", "init") {
    pid_t pid;

    constexpr int N = 1<<5;
    if (pid == 0) {
        // 子行程
        printf("Child process!n");
        void* d_x;
        cudaMalloc(&d_x, N*sizeof(float));
        sendSharedCache(d_x);

    } else if (pid > 0) {
        // 父行程
        void* d_y;
        recvSharedCache(d_x);
        printf("Parent process!n");
    } else {
        // 錯誤
        printf("Error!n");
    }

    return 0;
}


}  // namespace ipc
}  // namespace wxgpumemmgr
