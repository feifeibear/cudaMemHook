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

#include "cudahooker.hpp"
#include <cstring>
#include <set>
#include <iostream>
#include "cudakernels.h"

namespace wxgpumemmgr{

struct CudaHook::Impl {
};


CudaHook::CudaHook() : m_(std::make_unique<Impl>()) {};

CudaHook &CudaHook::instance()
{
    static CudaHook hook;
    return hook;
}

//bool CudaHook::IsValid(const char* symbol) const {
//    if (strcmp(symbol, "cuMemFree_v2") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) {
//        return true;
//    } else {
//        return false;
//    }
//}

void* CudaHook::GetFunction(const char* symbol) {
    if (strcmp(symbol, "cuMemFree_v2") == 0) {
        return reinterpret_cast<void *>(wx_cuMemFree_v2);
    } else if (strcmp(symbol, "cuMemAlloc_v2") == 0) {
        return reinterpret_cast<void *>(wx_cuMemAlloc_v2);
    } else {
        return nullptr;
    }
}

CudaHook::~CudaHook() = default;

} // namespace wxgpumemmgr
