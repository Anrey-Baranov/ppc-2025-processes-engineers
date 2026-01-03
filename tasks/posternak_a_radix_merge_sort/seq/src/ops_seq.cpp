#include "posternak_a_radix_merge_sort/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "posternak_a_radix_merge_sort/common/include/common.hpp"

namespace posternak_a_radix_merge_sort {

PosternakARadixMergeSortSEQ::PosternakARadixMergeSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PosternakARadixMergeSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool PosternakARadixMergeSortSEQ::PreProcessingImpl() {
  return true;
}

bool PosternakARadixMergeSortSEQ::RunImpl() {
  const std::vector<int> &input = GetInput();
  const auto n = static_cast<int>(input.size());

  std::vector<uint32_t> unsigned_data(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    unsigned_data[static_cast<size_t>(i)] = static_cast<uint32_t>(input[static_cast<size_t>(i)]) ^ 0x80000000U;
  }

  std::vector<uint32_t> buffer(static_cast<size_t>(n));

  constexpr int kNumBytes = 4;
  constexpr uint32_t kByteMask = 0xFFU;

  for (int byte_index = 0; byte_index < kNumBytes; byte_index++) {
    std::vector<int> count(256, 0);

    for (uint32_t value : unsigned_data) {
      const auto current_byte = static_cast<uint8_t>((value >> (byte_index * 8)) & kByteMask);
      ++count[current_byte];
    }

    for (int i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
      const auto current_byte =
          static_cast<uint8_t>((unsigned_data[static_cast<size_t>(i)] >> (byte_index * 8)) & kByteMask);
      buffer[static_cast<size_t>(--count[current_byte])] = unsigned_data[static_cast<size_t>(i)];
    }

    unsigned_data.swap(buffer);
  }

  std::vector<int> sorted_output(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    sorted_output[static_cast<size_t>(i)] = static_cast<int>(unsigned_data[static_cast<size_t>(i)] ^ 0x80000000U);
  }

  GetOutput() = std::move(sorted_output);
  return true;
}

bool PosternakARadixMergeSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace posternak_a_radix_merge_sort
