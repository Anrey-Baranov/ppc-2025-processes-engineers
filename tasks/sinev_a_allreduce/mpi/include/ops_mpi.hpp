#pragma once

#include <mpi.h>

#include "sinev_a_allreduce/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sinev_a_allreduce {

class SinevAAllreduce : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SinevAAllreduce(const InType &in);

  static void PerformOperation(void *inout, const void *in, int count, MPI_Datatype datatype, MPI_Op op);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int MpiAllreduceCustom(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                MPI_Comm comm);

  static int GetTypeSize(MPI_Datatype datatype);
};

}  // namespace sinev_a_allreduce
