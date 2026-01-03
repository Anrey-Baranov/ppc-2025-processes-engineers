#include "sinev_a_quicksort_with_simple_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <stack>
#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

SinevAQuicksortWithSimpleMergeSEQ::SinevAQuicksortWithSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool SinevAQuicksortWithSimpleMergeSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool SinevAQuicksortWithSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

int SinevAQuicksortWithSimpleMergeSEQ::Partition(std::vector<int> &arr, int left, int right) {
  int pivot_index = left + ((right - left) / 2);
  int pivot_value = arr[pivot_index];

  std::swap(arr[pivot_index], arr[left]);

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) {
      i++;
    }

    while (i <= j && arr[j] > pivot_value) {
      j--;
    }

    if (i < j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    } else {
      break;
    }
  }

  std::swap(arr[left], arr[j]);

  return j;
}

void SinevAQuicksortWithSimpleMergeSEQ::SimpleMerge(std::vector<int> &arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);

  int i = left;
  int j = mid + 1;
  int k = 0;  // Индекс для временного массива

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k] = arr[i];
      i++;
    } else {
      temp[k] = arr[j];
      j++;
    }
    k++;
  }

  while (i <= mid) {
    temp[k] = arr[i];
    i++;
    k++;
  }

  while (j <= right) {
    temp[k] = arr[j];
    j++;
    k++;
  }

  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void SinevAQuicksortWithSimpleMergeSEQ::QuickSortWithSimpleMerge(std::vector<int> &arr, int left, int right) {
  struct Range {
    int left;
    int right;
  };

  std::stack<Range> stack;
  stack.push({left, right});

  while (!stack.empty()) {
    Range current = stack.top();
    stack.pop();

    int l = current.left;
    int r = current.right;

    if (l >= r) {
      continue;
    }

    int pivot_index = Partition(arr, l, r);

    int left_size = pivot_index - l;
    int right_size = r - pivot_index;

    if (left_size > 1 && right_size > 1) {
      if (left_size > right_size) {
        stack.push({pivot_index + 1, r});
        stack.push({l, pivot_index - 1});
      } else {
        stack.push({l, pivot_index - 1});
        stack.push({pivot_index + 1, r});
      }
    } else if (left_size > 1) {
      stack.push({l, pivot_index - 1});
    } else if (right_size > 1) {
      stack.push({pivot_index + 1, r});
    }

    // Выполняем слияние
    SimpleMerge(arr, l, pivot_index, r);
  }
}

bool SinevAQuicksortWithSimpleMergeSEQ::RunImpl() {
  QuickSortWithSimpleMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);

  return true;
}

bool SinevAQuicksortWithSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_quicksort_with_simple_merge
