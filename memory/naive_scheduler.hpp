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
#include "ischeduler.hpp"
#include <memory>

namespace turbo_hook {
namespace memory {

class NaiveScheduler : public IScheduler {
public:
  explicit NaiveScheduler(size_t capacity);
  size_t Alloc(size_t size) override;
  void Free(size_t offset) override;
  void ShowList() const;
  ~NaiveScheduler();
  NaiveScheduler(NaiveScheduler &&o) = default;

private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

} // namespace memory
} // namespace turbo_hook
