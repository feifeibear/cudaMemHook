#pragma once

#include "alloc.grpc.pb.h"
#include "real_dlsym.h"
#include <grpcpp/grpcpp.h>

namespace turbo_hooker {
namespace service {

class CudaAllocServer : public CudaAllocator::Service {
public:
  using CudaMallocFn = int(void **, size_t);
  using CudaFreeFn = int(void *);
  CudaAllocServer();
  grpc::Status Malloc(grpc::ServerContext *context,
                      const MallocRequest *request,
                      ::MallocReply *response) override;

  grpc::Status Free(grpc::ServerContext *context, const FreeRequest *request,
                    FreeReply *response) override;

private:
  CudaMallocFn *cuda_malloc_;
  CudaFreeFn *cuda_free_;
};

} // namespace service
} // namespace turbo_hooker