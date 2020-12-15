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
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include "realdlsym.h"

using namespace wxgpumemmgr;

typedef int (*cuda_mem_alloc_v2_fp)(uintptr_t *, size_t);
typedef int (*cuda_mem_free_v2_fp)(uintptr_t);

extern "C" {

int wx_cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
    std::cerr << "call wx_cuMemAlloc_v2" << std::endl;
    int ret;

    static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2 = nullptr;
    if (orig_cuda_mem_alloc_v2 == nullptr)
        orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(get_dlopen_handle(),
                                                                                   "cuMemAlloc_v2"));

    ret = orig_cuda_mem_alloc_v2(devPtr, size);

    return ret;
}

int wx_cuMemAlloc(uintptr_t *devPtr, size_t size) {
    std::cerr << "call wx_cuMemAlloc" << std::endl;
    int ret;

    static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc = nullptr;
    if (orig_cuda_mem_alloc == nullptr)
        orig_cuda_mem_alloc = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(get_dlopen_handle(), "cuMemAlloc"));

    ret = orig_cuda_mem_alloc(devPtr, size);

    return ret;
}

int wx_cuMemFree_v2(uintptr_t ptr) {
    std::cerr << "call wx_cuMemFree_v2" << std::endl;
    static cuda_mem_free_v2_fp orig_cuda_mem_free_v2 = nullptr;
    if (orig_cuda_mem_free_v2 == nullptr)
        orig_cuda_mem_free_v2 = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(get_dlopen_handle(),
                                                                                 "cuMemFree_v2"));

    return orig_cuda_mem_free_v2(ptr);
}

int wx_cuMemFree(uintptr_t ptr) {
    std::cerr << "call wx_cuMemFree" << std::endl;
    static cuda_mem_free_v2_fp orig_cuda_mem_free = nullptr;
    if (orig_cuda_mem_free == nullptr)
        orig_cuda_mem_free = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(get_dlopen_handle(), "cuMemFree"));

    return orig_cuda_mem_free(ptr);
}

} // extern "C"
