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

#include <memory>

namespace turbo_hook {
namespace service {

class CudaAllocClient {
public:
  CudaAllocClient(const std::string &addr, uint16_t port);
  int Malloc(uintptr_t *ptr, size_t size);
  int Free(uintptr_t ptr);
  int Register(pid_t pid);
  uintptr_t uMalloc(pid_t pid, size_t size);
  void uFree(pid_t pid, uintptr_t addr);

private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

extern "C" {
extern int Malloc(uintptr_t *ptr, size_t size);
extern int Free(uintptr_t ptr);
extern int Register(pid_t pid);
extern void *Dlsym(void *handle, const char *symbol);
extern uintptr_t uMalloc(pid_t pid, size_t size);
extern void uFree(pid_t pid, uintptr_t addr);
}

} // namespace service
} // namespace turbo_hook
