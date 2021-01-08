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

#include "naive_scheduler.hpp"
#include <iostream>
#include <list>

namespace turbo_hook {
namespace memory {

struct MemoryBlock {
  size_t start_offset;
  size_t size;
};

struct NaiveScheduler::Impl {
  explicit Impl(size_t capacity) : capacity_(capacity) {}

  size_t capacity_;
  std::list<MemoryBlock> block_list_;
};

NaiveScheduler::NaiveScheduler(size_t capacity) : m_(new Impl(capacity)) {}

size_t NaiveScheduler::Alloc(size_t size) {
  size_t prev_end_offset = 0;
  for (auto it = m_->block_list_.begin(); it != m_->block_list_.end(); it++) {
    size_t start_offset = it->start_offset;
    size_t block_size = it->size;
    size_t gap = start_offset - prev_end_offset;
    if (gap >= size) {
      MemoryBlock new_block{prev_end_offset, size};
      m_->block_list_.insert(it, std::move(new_block));
      return prev_end_offset;
    }
    prev_end_offset = start_offset + block_size;
  }
  if (m_->capacity_ - prev_end_offset >= size) {
    MemoryBlock new_block{prev_end_offset, size};
    m_->block_list_.push_back(std::move(new_block));
    return prev_end_offset;
  }
  return -1U;
}

void NaiveScheduler::Free(size_t offset) {
  std::list<MemoryBlock>::iterator to_delete_it = m_->block_list_.end();
  for (auto it = m_->block_list_.begin(); it != m_->block_list_.end(); it++) {
    if (it->start_offset == offset) {
      to_delete_it = it;
      break;
    }
  }
  if (m_->block_list_.end() != to_delete_it) {
    m_->block_list_.erase(to_delete_it);
  } else {
    std::cerr << "Free an invalid memory offset " << offset << std::endl;
  }
}

void NaiveScheduler::ShowList() const {
  std::cerr << "ShowList" << std::endl;
  for (auto it = m_->block_list_.begin(); it != m_->block_list_.end(); it++) {
    std::cerr << it->start_offset << " " << it->size << std::endl;
  }
}

NaiveScheduler::~NaiveScheduler() = default;

} // namespace memory
} // namespace turbo_hook
