#pragma once

#include <dlfcn.h>

#ifdef __APPLE__
void *dlsym(void *handle, const char *symbol) __DYLDDL_DRIVERKIT_UNAVAILABLE;
#else
void *dlsym(void *handle, const char *symbol) noexcept;
#endif
