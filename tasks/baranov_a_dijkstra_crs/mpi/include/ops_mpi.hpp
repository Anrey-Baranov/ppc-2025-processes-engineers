#pragma once

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BaranovADijkstraCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeGraphData();
  void GatherResults();

  std::vector<int> local_offsets;
  std::vector<int> local_columns;
  std::vector<double> local_values;
  std::vector<int> vertex_ownership;
  int local_num_vertices;
  int world_size;
  int world_rank;
};

}  // namespace baranov_a_dijkstra_crs
