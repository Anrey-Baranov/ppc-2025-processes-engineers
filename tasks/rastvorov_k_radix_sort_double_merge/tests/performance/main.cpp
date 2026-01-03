#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rastvorov_k_radix_sort_double_merge/common/include/common.hpp"
#include "rastvorov_k_radix_sort_double_merge/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"

namespace rastvorov_k_radix_sort_double_merge {

namespace {

double NextRandDouble(uint32_t *state) {
  *state = (*state * 1103515245U) + 12345U;
  const uint32_t x = *state;
  const double u = static_cast<double>(x) / static_cast<double>(UINT32_MAX);
  return (u * 2.0 - 1.0) * 1.0e6;
}

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

class RastvorovKRadixSortDoubleMergeRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr std::size_t kSize = 200000;
    input_data_.resize(kSize);

    uint32_t state = 17U;
    for (std::size_t i = 0; i < kSize; ++i) {
      input_data_[i] = NextRandDouble(&state);
    }

    expected_ = input_data_;
    RadixSortDouble(&expected_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

TEST_P(RastvorovKRadixSortDoubleMergeRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

// MPI-only perf tasks (no seq_enabled)
const auto kAllPerfTasks = ppc::util::MakePerfTaskTuples<RastvorovKRadixSortDoubleMergeMPI, InType>(
    PPC_SETTINGS_rastvorov_k_radix_sort_double_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = RastvorovKRadixSortDoubleMergeRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RastvorovKRadixSortDoubleMergeRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace rastvorov_k_radix_sort_double_merge
