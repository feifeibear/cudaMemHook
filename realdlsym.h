#pragma once
#include <cstdint>

namespace wxgpumemmgr {

void *real_dlsym(void *handle, const char *symbol) noexcept;

template<typename FnPtrT>
constexpr auto func_cast(void *ptr) noexcept {
    return reinterpret_cast<FnPtrT>(reinterpret_cast<intptr_t>(ptr));
}

} // namespace wxgpumemmgr