#include "safronov_m_quicksort_with_batcher_even_odd_merge/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "safronov_m_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"

namespace safronov_m_quicksort_with_batcher_even_odd_merge {

SafronovMQuicksortWithBatcherEvenOddMergeSEQ::SafronovMQuicksortWithBatcherEvenOddMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SafronovMQuicksortWithBatcherEvenOddMergeSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool SafronovMQuicksortWithBatcherEvenOddMergeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::pair<int, int> SafronovMQuicksortWithBatcherEvenOddMergeSEQ::SplitRange(std::vector<int> &array, int left,
                                                                             int right) {
  int i = left;
  int j = right;
  int mid = left + ((right - left) / 2);
  int pivot = array[mid];

  while (i <= j) {
    while (array[i] < pivot) {
      i++;
    }
    while (array[j] > pivot) {
      j--;
    }
    if (i <= j) {
      int tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
      i++;
      j--;
    }
  }

  return {i, j};
}

bool SafronovMQuicksortWithBatcherEvenOddMergeSEQ::RunImpl() {
  std::vector<int> array = GetInput();
  if (array.empty()) {
    return true;
  }

  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(0, static_cast<int>(array.size()) - 1);

  while (!stack.empty()) {
    auto range = stack.back();
    stack.pop_back();

    int left = range.first;
    int right = range.second;

    if (left >= right) {
      continue;
    }

    auto borders = SplitRange(array, left, right);

    if (left < borders.second) {
      stack.emplace_back(left, borders.second);
    }
    if (borders.first < right) {
      stack.emplace_back(borders.first, right);
    }
  }

  GetOutput().swap(array);
  return true;
}

bool SafronovMQuicksortWithBatcherEvenOddMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace safronov_m_quicksort_with_batcher_even_odd_merge
