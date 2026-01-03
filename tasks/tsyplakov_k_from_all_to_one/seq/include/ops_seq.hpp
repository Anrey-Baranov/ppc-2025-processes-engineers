#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsyplakov_k_from_all_to_one/common/include/common.hpp"

namespace tsyplakov_k_from_all_to_one {

using InTypeSEQ = InTypeT<int>;
using OutTypeSEQ = OutTypeT<int>;
using BaseTaskSEQ = BaseTaskT<int>;

class TsyplakovKFromAllToOneSEQ : public BaseTaskSEQ {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit TsyplakovKFromAllToOneSEQ(const InTypeSEQ &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> gathered_;
};

}  // namespace tsyplakov_k_from_all_to_one
