#pragma once

#include "Rastvorov_K_Number_of_character_alternations/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rastvorov_k_number_of_character_alternations {

class RastvorovKNumberAfCharacterAlternationsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RastvorovKNumberAfCharacterAlternationsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rastvorov_k_number_of_character_alternations
