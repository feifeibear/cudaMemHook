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

#include "rpc/msgpack.hpp"
#include <sys/types.h>
namespace turbo_hook {
namespace service {

struct RegistRequest {
  pid_t pid_;
  MSGPACK_DEFINE_ARRAY(pid_);
};

struct RegistReply {
  uintptr_t original_ptr_;
  std::string ipc_handle_bytes_;
  MSGPACK_DEFINE_ARRAY(original_ptr_, ipc_handle_bytes_);
};

struct MallocRequest {
  size_t size_;
  MSGPACK_DEFINE_ARRAY(size_);
};

struct MallocReply {
  uintptr_t original_ptr_;
  std::string ipc_handle_bytes_;
  size_t offset_;
  MSGPACK_DEFINE_ARRAY(original_ptr_, ipc_handle_bytes_, offset_);
};

struct FreeRequest {
  uintptr_t original_ptr_;
  size_t offset_;
  MSGPACK_DEFINE_ARRAY(original_ptr_, offset_);
};

} // namespace service
} // namespace turbo_hook
