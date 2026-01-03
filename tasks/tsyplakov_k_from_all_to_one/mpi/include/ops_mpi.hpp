#pragma once

#include <mpi.h>

#include <vector>

#include "task/include/task.hpp"
#include "tsyplakov_k_from_all_to_one/common/include/common.hpp"

namespace tsyplakov_k_from_all_to_one {

template <typename T>
class TsyplakovKFromAllToOneMPI : public BaseTaskT<T> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TsyplakovKFromAllToOneMPI(const InTypeT<T> &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<T> gathered_;
};

int MyMpiGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm);

}  // namespace tsyplakov_k_from_all_to_one
