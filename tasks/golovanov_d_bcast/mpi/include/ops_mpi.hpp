#pragma once

#include <mpi.h>

#include "golovanov_d_bcast/common/include/common.hpp"
#include "task/include/task.hpp"
namespace golovanov_d_bcast {

class GolovanovDBcastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GolovanovDBcastMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static int MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
};

}  // namespace golovanov_d_bcast
