#include "rastvorov_k_radix_sort_double_merge/seq/include/ops_seq.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rastvorov_k_radix_sort_double_merge/common/include/common.hpp"

namespace rastvorov_k_radix_sort_double_merge {

namespace {

inline uint64_t DoubleToOrderedKey(double x) {
  if (std::isnan(x)) {
    return UINT64_MAX;
  }

  const auto bits = std::bit_cast<uint64_t>(x);

  const uint64_t sign = bits >> 63U;
  if (sign != 0U) {
    return ~bits;
  }
  return bits ^ (1ULL << 63U);
}

void RadixSortDouble(std::vector<double> *vec) {
  auto &a = *vec;
  const std::size_t n = a.size();
  if (n <= 1) {
    return;
  }

  std::vector<double> out(n);
  std::vector<uint64_t> keys(n);
  std::vector<uint64_t> out_keys(n);

  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = DoubleToOrderedKey(a[i]);
  }

  for (std::size_t pass = 0; pass < 8; ++pass) {
    std::array<std::size_t, 256> count{};
    const std::size_t shift = pass * 8;

    for (std::size_t i = 0; i < n; ++i) {
      const auto byte = static_cast<unsigned>((keys[i] >> shift) & 0xFFULL);
      count.at(static_cast<std::size_t>(byte))++;
    }

    std::array<std::size_t, 256> pos{};
    pos.at(0) = 0;
    for (std::size_t byte_idx = 1; byte_idx < pos.size(); ++byte_idx) {
      pos.at(byte_idx) = pos.at(byte_idx - 1) + count.at(byte_idx - 1);
    }

    for (std::size_t i = 0; i < n; ++i) {
      const auto byte = static_cast<unsigned>((keys[i] >> shift) & 0xFFULL);
      const std::size_t p = pos.at(static_cast<std::size_t>(byte))++;
      out[p] = a[i];
      out_keys[p] = keys[i];
    }

    a.swap(out);
    keys.swap(out_keys);
  }
}

}  // namespace

RastvorovKRadixSortDoubleMergeSEQ::RastvorovKRadixSortDoubleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RastvorovKRadixSortDoubleMergeSEQ::ValidationImpl() {
  return true;
}

bool RastvorovKRadixSortDoubleMergeSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool RastvorovKRadixSortDoubleMergeSEQ::RunImpl() {
  RadixSortDouble(&data_);
  return true;
}

bool RastvorovKRadixSortDoubleMergeSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace rastvorov_k_radix_sort_double_merge
