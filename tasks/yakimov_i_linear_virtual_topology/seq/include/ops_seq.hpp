#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"
#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"

namespace yakimov_i_linear_virtual_topology {

class YakimovILinearVirtualTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YakimovILinearVirtualTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void ReadDataFromFile(const std::string &filename);
  void ProcessDataSequentially();

  std::vector<int> data_;
  std::vector<int> process_values_;
  int total_sum_{0};
  std::string data_filename_;
};

}  // namespace yakimov_i_linear_virtual_topology
