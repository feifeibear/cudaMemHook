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
#include <dlfcn.h>

namespace wxgpumemmgr{

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

    void* m_header_{nullptr};
//
//    std::function<int(uintptr_t)> cuMemFree_v2{nullptr};
//    std::function<int(uintptr_t *, size_t)> cuMemAlloc_v2{nullptr};
    static int wx_cuMemFree_v2(uintptr_t addr) {
        std::cerr << "call wx_cuMemFree_v2" << std::endl;
        return 0;
    }

    static int wx_cuMemAlloc_v2(uintptr_t * addr, size_t size) {
        std::cerr << "call wx_cuMemAlloc_v2 " << size << std::endl;
        return 0;
    }
};

CudaHook::CudaHook(const char *dl) : m_(std::make_unique<Impl>()){
    // Load the libcuda.so library with RTLD_GLOBAL so we can hook the function calls
    m_->m_header_ = dlopen(dl, RTLD_LAZY | RTLD_GLOBAL);
    if (!m_->m_header_) {
        std::cerr << "Error to open library " << dl << ": " << dlerror() << std::endl;
        std::exit(-1);
    }

    // Load cuda APIs from libcuda.so
//    m_->cuMemFree_v2 = func_cast<int(uintptr_t)>(real_dlsym(m_->m_header_, "cuMemFree_v2"));
//    if (!m_->cuMemFree_v2) {
//        std::cerr << "Error to find symbol cuMemFree_v2 : " << dlerror() << std::endl;
//        std::exit(-2);
//    }
//    m_->cuMemAlloc_v2 = func_cast<int(uintptr_t *, size_t)>(real_dlsym(m_->m_header_, "cuMemAlloc_v2"));
//    if (!m_->cuMemAlloc_v2) {
//        std::cerr << "Error to find symbol cuMemAlloc_v2 : " << dlerror() << std::endl;
//        std::exit(-2);
//    }
}

CudaHook &CudaHook::instance()
{
    static CudaHook hook("libcuda.so");
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

CudaHook::~CudaHook() {
    if (m_->m_header_) {
        dlclose(m_->m_header_);
    }
}

} // namespace wxgpumemmgr
