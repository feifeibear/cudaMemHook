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

#include "cuda_alloc_client.h"
#include "cuda_runtime.h"
#include <iostream>
#include <memory>

int main(int argc, char **argv) {
  pid_t pid = 123;
  turbo_hook::service::Register(pid);
  uintptr_t dmem = turbo_hook::service::uMalloc(pid, 100);

  size_t len = 5;
  std::unique_ptr<int[]> hmem = std::make_unique<int[]>(len);
  for (int i = 0; i < len; ++i) {
    hmem[i] = i;
  }
  cudaMemcpy((void *)dmem, (void *)hmem.get(), len * sizeof(int),
             cudaMemcpyHostToDevice);

  std::unique_ptr<int[]> hmem_ref = std::make_unique<int[]>(len);
  cudaMemcpy((void *)hmem_ref.get(), (void *)dmem, len * sizeof(int),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < len; ++i) {
    std::cerr << hmem[i] << std::endl;
  }
  return 0;
}
