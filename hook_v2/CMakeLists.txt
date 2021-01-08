add_library(cuda_hook_v2 SHARED dlsym_hook.cpp real_dlsym.cpp)
target_link_libraries(cuda_hook_v2 PUBLIC cuda_alloc_client)
