#include "dlsym_hooker.h"
#include "real_dlsym.h"
#include <string>
#include <unordered_set>

extern "C" {
#ifdef __APPLE__
void *dlsym(void *handle, const char *symbol) __DYLDDL_DRIVERKIT_UNAVAILABLE {
#else
void *dlsym(void *handle, const char *symbol) noexcept {
#endif
  static auto *real_dlsym = turbo_hooker::GetRealDlsym();
  // TODO: CudaAllocClient
  static std::unordered_set<std::string> to_hook{"cuMemAlloc_v2",
                                                 "cuMemFree_v2"};
  auto *dlsym_func = to_hook.count(symbol) > 0 ? nullptr : real_dlsym;
  return (*dlsym_func)(handle, symbol);
}
}