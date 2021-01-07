#include "real_dlsym.h"
#include <cstdint>
#include <dlfcn.h>

extern "C" {
// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface
// function
void *__libc_dlsym(void *map, const char *name);
}

namespace turbo_hooker {
template <typename FnPtrT> constexpr auto func_cast(void *ptr) noexcept {
  return reinterpret_cast<FnPtrT>(reinterpret_cast<intptr_t>(ptr));
}

DlsymFn *GetRealDlsym() noexcept {
  return func_cast<DlsymFn *>(
      __libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym"));
}
}