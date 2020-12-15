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
#include <functional>
#include <memory>

namespace wxgpumemmgr {
/**
 * A sinlgeton maintains home-grown cuda APIs
 */
class CudaHook {
public:
    ~CudaHook();
    static CudaHook &instance();

    bool  IsValid(const char* symbol) const;
    void* GetFunction(const char* symbol);

private:
    CudaHook();
    struct Impl;
    std::unique_ptr<Impl> m_;
};

} // namespace wxgpumemmgr
