#include "belov_e_bubble_sort/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "belov_e_bubble_sort/common/include/common.hpp"

namespace belov_e_bubble_sort {

BelovEBubbleSortSEQ::BelovEBubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BelovEBubbleSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool BelovEBubbleSortSEQ::PreProcessingImpl() {
  return true;
}

bool BelovEBubbleSortSEQ::RunImpl() {
  std::vector<int> data = GetInput();
  size_t n = data.size();

  int temp = 0;

  std::vector<int> out;
  out.resize(n);
  for (size_t i = 0; i < n; out[n - i - 1] = data[n - i - 1], i++) {
    for (size_t j = 0; j < n - i - 1; j++) {
      if (data[j] > data[j + 1]) {
        temp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = temp;
      }
    }
  }
  GetOutput() = out;
  return true;
}

bool BelovEBubbleSortSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace belov_e_bubble_sort
