#pragma once

#include "pankov_gauss_filter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pankov_gauss_filter {

class PankovGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PankovGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pankov_gauss_filter
