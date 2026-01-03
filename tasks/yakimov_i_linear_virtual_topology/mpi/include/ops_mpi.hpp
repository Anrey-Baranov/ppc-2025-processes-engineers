#pragma once

#include <mpi.h>

#include <string>
#include <vector>

#include "task/include/task.hpp"
#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"

namespace yakimov_i_linear_virtual_topology {

class YakimovILinearVirtualTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit YakimovILinearVirtualTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void ReadDataFromFile(const std::string &filename);
  static void CreateLinearTopology(MPI_Comm &linear_comm);
  void ProcessDataInTopology(int rank, MPI_Comm &linear_comm);
  void ExchangeDataInTopology(MPI_Comm &linear_comm);

  std::vector<int> data_;
  int local_sum_{0};
  int total_sum_{0};
  std::string data_filename_;
};

}  // namespace yakimov_i_linear_virtual_topology
