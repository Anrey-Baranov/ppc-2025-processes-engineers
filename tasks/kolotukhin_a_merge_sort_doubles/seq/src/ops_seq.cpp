#include "kolotukhin_a_merge_sort_doubles/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "kolotukhin_a_merge_sort_doubles/common/include/common.hpp"

namespace kolotukhin_a_merge_sort_doubles {

namespace {

std::vector<std::uint64_t> ConvertDoublesToKeys(const std::vector<double> &data) {
  const std::size_t data_size = data.size();
  std::vector<std::uint64_t> keys(data_size);

  for (std::size_t i = 0; i < data_size; i++) {
    std::uint64_t u = 0;
    std::memcpy(&u, &data[i], sizeof(double));
    if ((u & 0x8000000000000000ULL) != 0U) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  return keys;
}

void ConvertKeysToDoubles(const std::vector<std::uint64_t> &keys, std::vector<double> &data) {
  const std::size_t data_size = data.size();
  for (std::size_t i = 0; i < data_size; i++) {
    std::uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0U) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data[i], &u, sizeof(double));
  }
}

void RadixSortUint64(std::vector<std::uint64_t> &keys) {
  const std::size_t data_size = keys.size();
  if (data_size <= 1) {
    return;
  }

  const int radix_size = 256;
  std::vector<std::uint64_t> temp(data_size);

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<std::size_t> count(radix_size + 1, 0);

    for (std::size_t i = 0; i < data_size; i++) {
      const auto digit = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFF);
      ++count[digit + 1];
    }

    for (int i = 0; i < radix_size; i++) {
      count[i + 1] += count[i];
    }

    for (std::size_t i = 0; i < data_size; i++) {
      const auto digit = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFF);
      const std::size_t pos = count[digit];
      temp[pos] = keys[i];
      count[digit] = pos + 1;
    }

    if (!temp.empty()) {
      for (std::size_t i = 0; i < data_size; i++) {
        keys[i] = temp[i];
      }
    }
  }
}

void RadixSortDoubles(std::vector<double> &data) {
  const std::size_t data_size = data.size();
  if (data_size <= 1) {
    return;
  }

  std::vector<std::uint64_t> keys = ConvertDoublesToKeys(data);

  RadixSortUint64(keys);

  ConvertKeysToDoubles(keys, data);
}

}  // namespace

KolotukhinAMergeSortDoublesSEQ::KolotukhinAMergeSortDoublesSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = std::vector<double>();
  std::get<1>(GetOutput()) = -1;
}

bool KolotukhinAMergeSortDoublesSEQ::ValidationImpl() {
  return true;
}

bool KolotukhinAMergeSortDoublesSEQ::PreProcessingImpl() {
  std::get<0>(GetOutput()).clear();
  return true;
}

bool KolotukhinAMergeSortDoublesSEQ::RunImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    std::get<0>(GetOutput()) = input;
    std::get<1>(GetOutput()) = 0;
    return true;
  }

  std::vector<double> data = input;
  RadixSortDoubles(data);

  std::get<0>(GetOutput()) = std::move(data);
  std::get<1>(GetOutput()) = 0;
  return true;
}

bool KolotukhinAMergeSortDoublesSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_merge_sort_doubles
