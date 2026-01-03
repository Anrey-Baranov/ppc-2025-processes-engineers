#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "zhurin_i_ring_topology/common/include/common.hpp"
#include "zhurin_i_ring_topology/mpi/include/ops_mpi.hpp"
#include "zhurin_i_ring_topology/seq/include/ops_seq.hpp"

namespace zhurin_i_ring_topology {

// Предварительное объявление функции PrintTo
static void PrintTo(const RingMessage &msg, ::std::ostream *os);

static void PrintTo(const RingMessage &msg, ::std::ostream *os) {
  *os << "RingMessage{source=" << msg.source << ", dest=" << msg.dest << ", data_size=" << msg.data.size()
      << ", go_clockwise=" << (msg.go_clockwise ? "true" : "false") << "}";
}

using TestType = std::tuple<int, RingMessage>;

class ZhurinIRingTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestName(
      const testing::TestParamInfo<std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>,
                                              std::string, TestType>> &info) {
    const auto &task_name = std::get<1>(info.param);
    const auto &test_param = std::get<2>(info.param);
    const int test_id = std::get<0>(test_param);
    return task_name + "Test" + std::to_string(test_id);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<2>(GetParam());
    test_message_ = std::get<1>(params);

    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized != 0) {
      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      const int max_node = std::max(test_message_.source, test_message_.dest);
      const int required_size = max_node + 1;
      if (size < required_size) {
        GTEST_SKIP() << "Test requires at least " << required_size << " processes, but only " << size << " available.";
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == test_message_.data;
  }

  InType GetTestInputData() final {
    return test_message_;
  }

 private:
  RingMessage test_message_;
};

namespace {

const std::array<TestType, 28> kAllTests = {
    std::make_tuple(1, RingMessage(0, 2, {1, 3, 5, 7, 9}, true)),
    std::make_tuple(2, RingMessage(7, 1, {111, 222, 333, 444}, true)),
    std::make_tuple(3, RingMessage(4, 4, {1, 2, 3, 4, 5}, true)),
    std::make_tuple(4, RingMessage(0, 6, {42, 52, 62, 72}, true)),
    std::make_tuple(5, RingMessage(2, 5, {}, true)),
    std::make_tuple(6, RingMessage(3, 7, {999}, true)),
    std::make_tuple(7, RingMessage(1, 1, {111}, true)),
    std::make_tuple(8, RingMessage(0, 3, {-1, -2, -3}, true)),
    std::make_tuple(9, RingMessage(5, 2, {0, 0, 0, 0}, true)),
    std::make_tuple(10, RingMessage(0, 1, {INT_MAX, INT_MIN}, true)),
    std::make_tuple(11, RingMessage(6, 0, {1, 2, 3, 4, 5, 6, 7}, true)),
    std::make_tuple(12, RingMessage(2, 2, std::vector<int>(100, 52), true)),
    std::make_tuple(13, RingMessage(0, 4, std::vector<int>(50, 5), true)),
    std::make_tuple(14, RingMessage(4, 0, {1, -2, 3, -4, 5, 6}, true)),
    std::make_tuple(15, RingMessage(0, 7, {1, 2, 3, 4, 5, 6, 7, 8}, true)),
    std::make_tuple(16, RingMessage(7, 0, {8, 7, 6, 5, 4, 3, 2, 1}, true)),
    std::make_tuple(17, RingMessage(0, 0, {1, 2, 3}, true)),
    std::make_tuple(18, RingMessage(0, 0, {7}, true)),
    std::make_tuple(19, RingMessage(0, 0, {-1, -2, -3}, true)),
    std::make_tuple(20, RingMessage(0, 0, {0, 0, 0}, true)),
    std::make_tuple(21, RingMessage(0, 2, {1, 3, 5, 7, 9}, false)),
    std::make_tuple(22, RingMessage(7, 1, {111, 222, 333, 444}, false)),
    std::make_tuple(23, RingMessage(0, 6, {42, 52, 62, 72}, false)),
    std::make_tuple(24, RingMessage(2, 5, {}, false)),
    std::make_tuple(25, RingMessage(3, 7, {999}, false)),
    std::make_tuple(26, RingMessage(0, 3, {-1, -2, -3}, false)),
    std::make_tuple(27, RingMessage(5, 2, {0, 0, 0, 0}, false)),
    std::make_tuple(28, RingMessage(6, 0, {1, 2, 3, 4, 5, 6, 7}, false)),
};

const auto kAllTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<zhurin_i_ring_topology::ZhurinIRingTopologyMPI, InType>(
                       kAllTests, PPC_SETTINGS_zhurin_i_ring_topology),
                   ppc::util::AddFuncTask<zhurin_i_ring_topology::ZhurinIRingTopologySEQ, InType>(
                       kAllTests, PPC_SETTINGS_zhurin_i_ring_topology));

inline const auto kGtestValues = ppc::util::ExpandToValues(kAllTasksList);

TEST_P(ZhurinIRingTopologyFuncTests, AllTests) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ZhurinIRingTopology, ZhurinIRingTopologyFuncTests, kGtestValues,
                         ZhurinIRingTopologyFuncTests::PrintTestName);

}  // namespace

}  // namespace zhurin_i_ring_topology
