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

#include "alloc.grpc.pb.h"
#include "real_dlsym.h"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace turbo_hooker {
namespace service {

class CudaAllocClient {
public:
  CudaAllocClient(const std::string &server_address);
  uintptr_t Malloc(size_t size);

  void Free(uintptr_t ptr);

private:
  std::unique_ptr<CudaAllocator::Stub> stub_;
  std::unordered_map<uintptr_t, Allocation> allocations_;
  std::mutex mtx_;
};

extern "C" {
extern int Malloc(uintptr_t *ptr, size_t size);
extern int Free(uintptr_t ptr);
extern void *Dlsym(void *handle, const char *symbol);
}

} // namespace service
} // namespace turbo_hooker
