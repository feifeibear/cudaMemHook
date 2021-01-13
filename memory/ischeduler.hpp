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
#include <cstdlib>

namespace turbo_hook {
namespace memory {

class IScheduler {
public:
  // @params size: the size of memory to be allocated
  // @ret: the offset of the memory
  virtual size_t Alloc(size_t size) = 0;

  // @params: offset: the memory offset to be freed
  virtual void Free(size_t offset) = 0;
  ~IScheduler();
};

} // namespace memory
} // namespace turbo_hook
