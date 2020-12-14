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
#include "dlsym_hook.h"
#include <set>
#include <iostream>
namespace wxgpumemmgr{
//TODO substitude it with API of a global cuda allocator.

namespace details {
    static int wx_cuMemFree_v2(uintptr_t addr) {
        std::cerr << "call wx_cuMemFree_v2" << std::endl;
        return 0;
    }

    static int wx_cuMemAlloc_v2(uintptr_t * addr, size_t size) {
        std::cerr << "call wx_cuMemAlloc_v2 " << size << std::endl;
        return 0;
    }
}

struct CudaHook::Impl {

    std::set<const char*> hanlder_names_{"cuMemFree_v2", "cuMemAlloc_v2"};

    static int wx_cuMemFree_v2(uintptr_t addr) {
        std::cerr << "call wx_cuMemFree_v2" << std::endl;
        return 0;
    }

    static int wx_cuMemAlloc_v2(uintptr_t * addr, size_t size) {
        std::cerr << "call wx_cuMemAlloc_v2 " << size << std::endl;
        return 0;
    }
};


CudaHook &CudaHook::instance()
{
    static CudaHook hook;
    return hook;
}

bool  CudaHook::IsValid(const char* symbol) const {
    return m_->hanlder_names_.count(symbol) == 0;
}

void* CudaHook::GetFunction(const char* symbol) {
    if (strcmp(symbol, "cuMemFree_v2")) {
        return reinterpret_cast<void *>(Impl::wx_cuMemFree_v2);
    } else if (strcmp(symbol, "cuMemAlloc_v2")) {
        return reinterpret_cast<void *>(Impl::wx_cuMemAlloc_v2);
    } else {
        std::cerr << "CudaHook GetFunction's parameter is invalid" << std::endl;
        std::exit(-2);
    }
}

CudaHook::~CudaHook() = default;

} // namespace wxgpumemmgr
