#pragma once

#include <mpi.h>

#include <iostream>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaranovADijkstraCrsMPI(const InType &in);

  ~BaranovADijkstraCrsMPI() noexcept override;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  template <typename T>
  std::vector<T> DijkstraParallelTemplate(int vertices, const std::vector<int> &row_ptr,
                                          const std::vector<int> &col_idx, const std::vector<T> &values, int source);

  bool ValidateGraph(const GraphCRS &graph);
  template <typename T>
  void TreeBroadcast(std::vector<T> &data, int root, MPI_Datatype mpi_type);

  template <typename T>
  void TreeAllReduceMin(std::vector<T> &data, MPI_Datatype mpi_type);
};

}  // namespace baranov_a_dijkstra_crs
