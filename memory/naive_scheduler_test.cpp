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

#include "catch2/catch.hpp"
#include "naive_scheduler.hpp"
#include <iostream>

namespace turbo_hook {
namespace memory {

TEST_CASE("naive_scheduler", "test1") {
  NaiveScheduler scheduler(100);

  auto offset = scheduler.Alloc(10);
  REQUIRE(offset == 0);
  offset = scheduler.Alloc(20);
  REQUIRE(offset == 10);

  scheduler.Free(0);
  offset = scheduler.Alloc(5);
  REQUIRE(offset == 0);
}

} // namespace memory
} // namespace turbo_hook