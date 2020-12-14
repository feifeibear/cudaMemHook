#pragma once
#include <dlfcn.h>
#include <cstring>
#include <stdio.h>
#include "realdlsym.h"

extern "C" {

int cuMemFree_v2(uintptr_t ptr) {
    printf("cuMemFree_v2\n");
    return 0;
}

int cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
    printf("cuMemAlloc_v2\n");
    return 0;
}

#if __APPLE__
void *dlsym(void * handle, const char * symbol) __DYLDDL_DRIVERKIT_UNAVAILABLE {
#else
void *dlsym(void *handle, const char *symbol) noexcept {
#endif
    printf("dlsym loading %s\n", symbol);
    if (strncmp(symbol, "cu", 2) != 0) {
        return wxgpumemmgr::real_dlsym(handle, symbol);
    } else if (strcmp(symbol, "cuMemAlloc_v2") == 0) {
        return reinterpret_cast<void *>(cuMemFree_v2);
    } else if (strcmp(symbol, "cuMemFree_v2") == 0) {
        return reinterpret_cast<void *>(cuMemAlloc_v2);
    }
    return wxgpumemmgr::real_dlsym(handle, symbol);
}

} // extern "C"

