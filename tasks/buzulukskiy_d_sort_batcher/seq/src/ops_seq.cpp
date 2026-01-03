#include "buzulukskiy_d_sort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"

namespace buzulukskiy_d_sort_batcher {
namespace {
constexpr int kRadixBase = 10;
constexpr std::size_t kBlockSize = 64;

void RadixSortUnsigned(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }
  const int max_val = *std::ranges::max_element(arr);
  std::vector<int> output(arr.size());
  for (int exp = 1; max_val / exp > 0; exp *= kRadixBase) {
    std::vector<int> count(static_cast<std::size_t>(kRadixBase), 0);
    for (const int val : arr) {
      count[static_cast<std::size_t>((val / exp) % kRadixBase)]++;
    }
    for (std::size_t i = 1; i < static_cast<std::size_t>(kRadixBase); ++i) {
      count[i] += count[i - 1];
    }
    for (std::size_t i = arr.size(); i-- > 0;) {
      const int digit = (arr[i] / exp) % kRadixBase;
      output[--count[static_cast<std::size_t>(digit)]] = arr[i];
    }
    arr.swap(output);
  }
}

void RadixSortLSD(std::vector<int> &data) {
  if (data.empty()) {
    return;
  }
  std::vector<int> positives;
  std::vector<int> negatives;
  for (const int value : data) {
    if (value < 0) {
      negatives.push_back(value == INT_MIN ? INT_MAX : -value);
    } else {
      positives.push_back(value);
    }
  }
  if (!positives.empty()) {
    RadixSortUnsigned(positives);
  }
  if (!negatives.empty()) {
    RadixSortUnsigned(negatives);
    std::ranges::reverse(negatives);
    for (int &v_ref : negatives) {
      v_ref = (v_ref == INT_MAX ? INT_MIN : -v_ref);
    }
  }
  data.assign(negatives.begin(), negatives.end());
  data.insert(data.end(), positives.begin(), positives.end());
}

void BatcherStep(std::vector<int> &data, std::size_t i, std::size_t j, std::size_t k, std::size_t phase_step) {
  const std::size_t r1 = i + j;
  const std::size_t r2 = i + j + k;
  if (r2 < data.size() && (r1 / (phase_step * 2)) == (r2 / (phase_step * 2))) {
    if (data[r1] > data[r2]) {
      std::swap(data[r1], data[r2]);
    }
  }
}

void BatcherInner(std::vector<int> &data, std::size_t k, std::size_t phase_step) {
  for (std::size_t j = k % phase_step; j + k < data.size(); j += 2 * k) {
    for (std::size_t i = 0; i < k; ++i) {
      BatcherStep(data, i, j, k, phase_step);
    }
  }
}

void BatcherMergeNetwork(std::vector<int> &data) {
  const std::size_t n = data.size();
  for (std::size_t phase_step = 1; phase_step < n; phase_step <<= 1) {
    for (std::size_t k = phase_step; k > 0; k >>= 1) {
      BatcherInner(data, k, phase_step);
    }
  }
}
}  // namespace

BuzulukskiyDSortBatcherSEQ::BuzulukskiyDSortBatcherSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BuzulukskiyDSortBatcherSEQ::ValidationImpl() {
  return true;
}
bool BuzulukskiyDSortBatcherSEQ::PreProcessingImpl() {
  return true;
}
bool BuzulukskiyDSortBatcherSEQ::PostProcessingImpl() {
  return true;
}

bool BuzulukskiyDSortBatcherSEQ::RunImpl() {
  if (GetInput().empty()) {
    GetOutput() = InType();
    return true;
  }
  std::vector<int> data = GetInput();
  for (std::size_t i = 0; i < data.size(); i += kBlockSize) {
    std::size_t current_size = std::min(kBlockSize, data.size() - i);
    std::vector<int> block(data.begin() + static_cast<std::ptrdiff_t>(i),
                           data.begin() + static_cast<std::ptrdiff_t>(i + current_size));
    RadixSortLSD(block);
    std::ranges::copy(block, data.begin() + static_cast<std::ptrdiff_t>(i));
  }
  BatcherMergeNetwork(data);
  GetOutput() = std::move(data);
  return true;
}
}  // namespace buzulukskiy_d_sort_batcher
