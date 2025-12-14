#pragma once

#include <mpi.h>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaranovADijkstraCrsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  template <typename T>
  std::vector<T> DijkstraParallelTemplate(int vertices, const std::vector<int> &row_ptr,
                                          const std::vector<int> &col_idx, const std::vector<T> &values, int source);

  template <typename T>
  void TreeBroadcast(std::vector<T> &data, int count, int root, MPI_Datatype datatype, MPI_Comm comm);

  template <typename T>
  void TreeAllReduceMin(std::vector<T> &local_data, std::vector<T> &global_data, int count, MPI_Datatype datatype,
                        MPI_Comm comm);
};

}  // namespace baranov_a_dijkstra_crs
