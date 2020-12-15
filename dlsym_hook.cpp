#include "dlsym_hook.h"
#include "cudahooker.hpp"
#include "realdlsym.h"

extern "C" {
#ifdef __APPLE__
void *dlsym(void * handle, const char * symbol) __DYLDDL_DRIVERKIT_UNAVAILABLE {
#else
void *dlsym(void *handle, const char *symbol) noexcept {
#endif
    auto& hooker = wxgpumemmgr::CudaHook::instance();
    if (hooker.IsValid(symbol)) {
    	printf("hooking %s\n", symbol);
        return hooker.GetFunction(symbol);
    }
    printf("dlsym loading %s\n", symbol);
    return wxgpumemmgr::real_dlsym(handle, symbol);
}
} // extern "C"
