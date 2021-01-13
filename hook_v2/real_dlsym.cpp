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

#include "real_dlsym.h"
#include <cstdint>
#include <dlfcn.h>

extern "C" {
// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
// function
void *__libc_dlsym(void *map, const char *name);
}

namespace turbo_hooker {
template <typename FnPtrT> constexpr auto func_cast(void *ptr) noexcept {
  return reinterpret_cast<FnPtrT>(reinterpret_cast<intptr_t>(ptr));
}

DlsymFn *GetRealDlsym() noexcept {
  return func_cast<DlsymFn *>(
      __libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym"));
}
} // namespace turbo_hooker
