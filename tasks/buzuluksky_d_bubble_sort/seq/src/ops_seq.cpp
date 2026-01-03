#include "buzuluksky_d_bubble_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "buzuluksky_d_bubble_sort/common/include/common.hpp"

namespace buzuluksky_d_bubble_sort {

BuzulukskyDBubbleSortSEQ::BuzulukskyDBubbleSortSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  static_cast<void>(GetOutput());
}

bool BuzulukskyDBubbleSortSEQ::ValidationImpl() {
  return true;
}
bool BuzulukskyDBubbleSortSEQ::PreProcessingImpl() {
  return true;
}

bool BuzulukskyDBubbleSortSEQ::RunImpl() {
  auto data = GetInput();
  const std::size_t n = data.size();
  if (n <= 1U) {
    GetOutput() = data;
    return true;
  }

  bool sorted = false;
  for (std::size_t pass = 0; pass < n && !sorted; ++pass) {
    sorted = true;
    for (std::size_t i = 0; i + 1 < n; i += 2) {
      if (data[i] > data[i + 1]) {
        std::swap(data[i], data[i + 1]);
        sorted = false;
      }
    }
    for (std::size_t i = 1; i + 1 < n; i += 2) {
      if (data[i] > data[i + 1]) {
        std::swap(data[i], data[i + 1]);
        sorted = false;
      }
    }
  }

  GetOutput() = data;
  return true;
}

bool BuzulukskyDBubbleSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace buzuluksky_d_bubble_sort
