#pragma once

#include <mpi.h>

#include <vector>

#include "belov_e_bubble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace belov_e_bubble_sort {
class BelovEBubbleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BelovEBubbleSortMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
void BubbleSort(std::vector<int> &arr);
std::vector<int> LeftMerge(const std::vector<int> &left, const std::vector<int> &right);
std::vector<int> RightMerge(const std::vector<int> &left, const std::vector<int> &right);
void LeftProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
                 MPI_Comm comm);
void RightProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
                  MPI_Comm comm);
void EvenPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
               MPI_Comm comm);
void OddPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
              MPI_Comm comm);
}  // namespace belov_e_bubble_sort
