#include "kopilov_d_shell_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "kopilov_d_shell_merge/common/include/common.hpp"

namespace kopilov_d_shell_merge {

namespace {

void ShellSort(std::vector<int> &vec) {
  const std::size_t n = vec.size();
  if (n < 2) {
    return;
  }
  for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = vec[i];
      std::size_t j = i;
      while (j >= gap && vec[j - gap] > tmp) {
        vec[j] = vec[j - gap];
        j -= gap;
      }
      vec[j] = tmp;
    }
  }
}

std::vector<int> SimpleMerge(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::ranges::merge(a, b, std::back_inserter(result));
  return result;
}

}  // namespace

KopilovDShellMergeSEQ::KopilovDShellMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KopilovDShellMergeSEQ::ValidationImpl() {
  return true;
}

bool KopilovDShellMergeSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool KopilovDShellMergeSEQ::RunImpl() {
  if (data_.size() < 2) {
    return true;
  }

  const auto mid = data_.size() / 2;
  std::vector<int> left(data_.begin(), data_.begin() + static_cast<std::ptrdiff_t>(mid));
  std::vector<int> right(data_.begin() + static_cast<std::ptrdiff_t>(mid), data_.end());

  ShellSort(left);
  ShellSort(right);

  data_ = SimpleMerge(left, right);
  return true;
}

bool KopilovDShellMergeSEQ::PostProcessingImpl() {
  GetOutput() = std::move(data_);
  return true;
}

}  // namespace kopilov_d_shell_merge
