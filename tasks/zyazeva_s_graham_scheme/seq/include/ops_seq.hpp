#pragma once

#include "task/include/task.hpp"
#include "zyazeva_s_graham_scheme/common/include/common.hpp"

namespace zyazeva_s_graham_scheme {

class ZyazevaSGrahamSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyazevaSGrahamSchemeSEQ(const InType &in);  // 2

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyazeva_s_graham_scheme
