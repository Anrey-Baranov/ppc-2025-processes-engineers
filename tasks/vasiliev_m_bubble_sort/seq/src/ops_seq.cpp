#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

VasilievMBubbleSortSEQ::VasilievMBubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBubbleSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMBubbleSortSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMBubbleSortSEQ::RunImpl() {
  auto &vec = GetInput();
  const size_t n = vec.size();
  bool sorted = false;

  if (vec.empty()) {
    return true;
  }

  while (!sorted) {
    sorted = true;

    for (size_t i = 0; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }
  }

  GetOutput() = vec;
  return true;
}

bool VasilievMBubbleSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bubble_sort
