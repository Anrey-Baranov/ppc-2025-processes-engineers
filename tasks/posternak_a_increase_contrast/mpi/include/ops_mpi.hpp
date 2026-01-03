#pragma once

#include <vector>

#include "posternak_a_increase_contrast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace posternak_a_increase_contrast {

class PosternakAIncreaseContrastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PosternakAIncreaseContrastMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<unsigned char> ScatterInputData(int rank, int size, int data_len);
  static void FindGlobalMinMax(const std::vector<unsigned char> &proc_part, unsigned char *data_min,
                               unsigned char *data_max);
  static std::vector<unsigned char> ApplyContrast(const std::vector<unsigned char> &proc_part, unsigned char data_min,
                                                  unsigned char data_max);
};

}  // namespace posternak_a_increase_contrast
