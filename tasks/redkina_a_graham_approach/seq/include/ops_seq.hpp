#pragma once

#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

class RedkinaAGrahamApproachSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaAGrahamApproachSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> GrahamScan(std::vector<Point> points);
};

}  // namespace redkina_a_graham_approach
