#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "zyazeva_s_graham_scheme/common/include/common.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZyazevaSGrahamSchemeMPI(const InType &in);  // 4

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> local_points_;
};

}  // namespace zyazeva_s_graham_scheme
