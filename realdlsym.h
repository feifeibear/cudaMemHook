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
#include <cstdint>

namespace wxgpumemmgr {

extern void *real_dlsym(void *handle, const char *symbol) noexcept;
//extern void *dlopen(const char *filename, int flags);
//extern void *get_dlopen_handle();

template<typename FnPtrT>
constexpr auto func_cast(void *ptr) noexcept {
    return reinterpret_cast<FnPtrT>(reinterpret_cast<intptr_t>(ptr));
}

} // namespace wxgpumemmgr
