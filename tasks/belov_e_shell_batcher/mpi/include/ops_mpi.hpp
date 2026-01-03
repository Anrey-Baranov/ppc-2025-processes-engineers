#pragma once

#include <mpi.h>

#include <vector>

#include "belov_e_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace belov_e_shell_batcher {
class BelovEShellBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BelovEShellBatcherMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
void ShellSort(std::vector<int> &arr);
std::vector<int> BatcherLeftMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                  int local_arr_size, int rank, MPI_Comm comm);
std::vector<int> BatcherRightMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                   int local_arr_size, int rank, MPI_Comm comm);
void LeftProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm);
void RightProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm);
void EvenPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm);
void OddPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm);
}  // namespace belov_e_shell_batcher
