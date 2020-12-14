#include "realdlsym.h"

#include <dlfcn.h>
extern "C" {
// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface function
void* __libc_dlsym (void *map, const char *name);
}

namespace wxgpumemmgr {

using FnDlsym = void *(void*, const char*);
void* real_dlsym(void *handle, const char* symbol) noexcept
{
    static auto internal_dlsym = func_cast<FnDlsym*>(__libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym"));
    return (*internal_dlsym)(handle, symbol);
}

} // namespace wxgpumemmgr