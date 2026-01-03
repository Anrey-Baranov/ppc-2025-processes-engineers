#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tochilin_e_vertical_ribbon_scheme/common/include/common.hpp"

namespace tochilin_e_vertical_ribbon_scheme {

class TochilinEVerticalRibbonSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TochilinEVerticalRibbonSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
  int rows_{0};
  int cols_{0};
};

}  // namespace tochilin_e_vertical_ribbon_scheme
