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
#include <dlfcn.h>
#include <cstring>
#include <stdio.h>
#include "realdlsym.h"
#include "cudahooker.hpp"
namespace wxgpumemmgr{
class CudaHook;
}


extern "C" {

#ifdef __APPLE__
void *dlsym(void * handle, const char * symbol) __DYLDDL_DRIVERKIT_UNAVAILABLE {
#else
void *dlsym(void *handle, const char *symbol) noexcept {
#endif
    printf("dlsym loading %s\n", symbol);
    auto& hooker = wxgpumemmgr::CudaHook::instance();
    if (hooker.IsValid(symbol)) {
        return hooker.GetFunction(symbol);
    }
    return wxgpumemmgr::real_dlsym(handle, symbol);
}

} // extern "C"
