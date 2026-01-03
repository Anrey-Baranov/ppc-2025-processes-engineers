#include "belov_e_shell_batcher/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "belov_e_shell_batcher/common/include/common.hpp"

namespace belov_e_shell_batcher {

BelovEShellBatcherSEQ::BelovEShellBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BelovEShellBatcherSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool BelovEShellBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool BelovEShellBatcherSEQ::RunImpl() {
  std::vector<int> data = GetInput();
  size_t n = data.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = data[i];
      size_t j = i;

      while (j >= gap && data[j - gap] > temp) {
        data[j] = data[j - gap];
        j -= gap;
      }

      data[j] = temp;
    }
  }

  GetOutput() = data;
  return true;
}

bool BelovEShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace belov_e_shell_batcher
