#if 0  // NOLINT(readability-avoid-unconditional-preprocessor-if)
#  include <gtest/gtest.h>
#  include <mpi.h>

#  include <array>
#  include <cstddef>
#  include <cstdint>
#  include <random>
#  include <stdexcept>
#  include <string>
#  include <tuple>
#  include <vector>

#  include "kondrashova_v_gauss_filter_vertical_split/common/include/common.hpp"
#  include "kondrashova_v_gauss_filter_vertical_split/mpi/include/ops_mpi.hpp"
#  include "kondrashova_v_gauss_filter_vertical_split/seq/include/ops_seq.hpp"
#  include "util/include/func_test_util.hpp"
#  include "util/include/util.hpp"

namespace kondrashova_v_gauss_filter_vertical_split {

namespace {

unsigned int GetSynchronizedSeed() {
  std::random_device rd;
  unsigned int seed = rd();

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized != 0) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      seed = 0;
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  }

  return seed;
}

}  // namespace

class KondrashovaVRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_.width = size;
    input_data_.height = size;
    input_data_.channels = 3;
    input_data_.pixels.resize(static_cast<size_t>(size) * size * 3);

    unsigned int seed = GetSynchronizedSeed();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    for (auto &pixel : input_data_.pixels) {
      pixel = static_cast<uint8_t>(dist(gen));
    }

    KondrashovaVGaussFilterVerticalSplitSEQ seq_task(input_data_);
    if (!seq_task.Validation() || !seq_task.PreProcessing() || !seq_task.Run() || !seq_task.PostProcessing()) {
      throw std::runtime_error("Failed to compute reference result with SEQ version");
    }
    expected_output_ = seq_task.GetOutput();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KondrashovaVRunFuncTestsProcesses, GaussFilterGenerated) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "small_10x10"), std::make_tuple(50, "medium_50x50"),
                                            std::make_tuple(100, "large_100x100")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KondrashovaVGaussFilterVerticalSplitMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kondrashova_v_gauss_filter_vertical_split),
                                           ppc::util::AddFuncTask<KondrashovaVGaussFilterVerticalSplitSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kondrashova_v_gauss_filter_vertical_split));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KondrashovaVRunFuncTestsProcesses::PrintFuncTestName<KondrashovaVRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussFilterTests, KondrashovaVRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kondrashova_v_gauss_filter_vertical_split

#endif
