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

#include "realdlsym.h"
#include <stdio.h>

#include <dlfcn.h>
extern "C" {
// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface function
void* __libc_dlsym (void *map, const char *name);
extern void *_dl_sym(void *, const char *, void *);
}

namespace wxgpumemmgr {

using FnDlsym = void *(void*, const char*);

using FnDlopen = void *(const char*, int);


void* real_dlsym(void *handle, const char* symbol) noexcept
{
    static auto internal_dlsym = func_cast<FnDlsym*>(__libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym"));
    return (*internal_dlsym)(handle, symbol);
}
/*
//FnDlopen* real_dlopen = nullptr;
void* dlopen_handle = nullptr;

void *dlopen(const char *filename, int flags) {
    static auto* real_dlopen = func_cast<FnDlopen*>(__libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlopen"));
    dlopen_handle = real_dlopen(filename, flags);
    return dlopen_handle;
}

void *get_dlopen_handle() {
    return dlopen_handle;
}
*/

} // namespace wxgpumemmgr
