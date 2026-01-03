#include "dorogin_v_bitwise_simple_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dorogin_v_bitwise_simple_merge/common/include/common.hpp"

namespace dorogin_v_bitwise_simple_merge {

namespace {

uint64_t DoubleToSortableKey(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  if ((bits >> 63) != 0) {
    bits = ~bits;
  } else {
    bits ^= (static_cast<uint64_t>(1) << 63);
  }
  return bits;
}

double SortableKeyToDouble(uint64_t bits) {
  if ((bits >> 63) != 0) {
    bits ^= (static_cast<uint64_t>(1) << 63);
  } else {
    bits = ~bits;
  }
  double value = 0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

void RadixSort(std::vector<double> &data) {
  if (data.size() <= 1) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    keys[i] = DoubleToSortableKey(data[i]);
  }

  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(257, 0);

    for (const auto &key : keys) {
      size_t bucket = (key >> shift) & 0xFF;
      count[bucket + 1]++;
    }

    for (int i = 1; i <= 256; i++) {
      count[i] += count[i - 1];
    }

    for (const auto &key : keys) {
      size_t bucket = (key >> shift) & 0xFF;
      temp[count[bucket]++] = key;
    }

    std::swap(keys, temp);
  }

  for (size_t i = 0; i < data.size(); i++) {
    data[i] = SortableKeyToDouble(keys[i]);
  }
}

}  // namespace

DoroginVBitwiseSimpleMergeSEQ::DoroginVBitwiseSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DoroginVBitwiseSimpleMergeSEQ::ValidationImpl() {
  return true;
}

bool DoroginVBitwiseSimpleMergeSEQ::PreProcessingImpl() {
  input_data_ = GetInput();
  output_data_.clear();
  return true;
}

bool DoroginVBitwiseSimpleMergeSEQ::RunImpl() {
  output_data_ = input_data_;
  RadixSort(output_data_);
  return true;
}

bool DoroginVBitwiseSimpleMergeSEQ::PostProcessingImpl() {
  GetOutput() = output_data_;
  return true;
}

}  // namespace dorogin_v_bitwise_simple_merge
