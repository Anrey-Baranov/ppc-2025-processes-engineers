#include <gtest/gtest.h>

#include <vector>

#include "../../common/include/common.hpp"
#include "../../seq/include/ops_seq.hpp"

#ifdef USE_MPI
#  include <mpi.h>
#endif

namespace tsyplakov_k_from_all_to_one {

TEST(TsyplakovKFromAllToOneMPI, GatherIntRoot0) {
#ifdef USE_MPI
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int local_size = 4;
  std::vector<int> local_vec(local_size, rank);
  int root = 0;

  InTypeT<int> in{local_vec, root};
  TsyplakovKFromAllToOneMPI<int> task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == root) {
    const auto &out = task.GetOutput();
    ASSERT_EQ(out.size(), size * local_size);

    for (int r = 0; r < size; ++r) {
      for (int i = 0; i < local_size; ++i) {
        EXPECT_EQ(out[r * local_size + i], r);
      }
    }
  }
#endif
}

TEST(TsyplakovKFromAllToOneMPI, GatherIntRootMiddle) {
#ifdef USE_MPI
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int root = size / 2;
  std::vector<int> local_vec(3, rank);

  InTypeT<int> in{local_vec, root};
  TsyplakovKFromAllToOneMPI<int> task(in);

  task.Run();

  if (rank == root) {
    const auto &out = task.GetOutput();
    ASSERT_EQ(out.size(), size * 3);
    for (int r = 0; r < size; ++r) {
      for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(out[r * 3 + i], r);
      }
    }
  }
#endif
}

TEST(TsyplakovKFromAllToOneSEQ, BasicSeq) {
  std::vector<int> data = {1, 2, 3, 4};
  int root = 0;

  InTypeSEQ in{data, root};
  TsyplakovKFromAllToOneSEQ task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &out = task.GetOutput();
  EXPECT_EQ(out, data);
}

}  // namespace tsyplakov_k_from_all_to_one
