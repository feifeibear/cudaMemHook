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
#include <unistd.h>
#include <stdint.h>


using cuda_mem_alloc_v2_fn = int(uintptr_t *, size_t);
using cuda_mem_free_v2_fn = int(uintptr_t);

extern "C" {

extern int wx_cuMemAlloc_v2(uintptr_t *devPtr, size_t size);
extern int wx_cuMemFree_v2(uintptr_t ptr);
extern int wx_cuMemAlloc(uintptr_t *devPtr, size_t size);
extern int wx_cuMemFree(uintptr_t ptr);

} // extern "C"
