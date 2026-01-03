#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

#include "buzuluksky_d_bubble_sort/mpi/include/ops_mpi.hpp"
#include "buzuluksky_d_bubble_sort/seq/include/ops_seq.hpp"

namespace buzuluksky_d_bubble_sort {

static bool IsSorted(const std::vector<int> &arr) {
  if (arr.size() <= 1) {
    return true;
  }
  for (std::size_t i = 0; i + 1 < arr.size(); ++i) {
    if (arr[i] > arr[i + 1]) {
      return false;
    }
  }
  return true;
}

class BubbleSortPerfTest : public ::testing::Test {
 protected:
  std::size_t size{5000};
  std::vector<int> input;

  void SetUp() override {
    input.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
      input[i] = static_cast<int>(i);
      if (i % 100 == 0) {
        input[i] = static_cast<int>((i * 37) % size);
      }
    }

    if (size > 10) {
      std::swap(input[0], input[size - 1]);
      std::swap(input[size / 4], input[size / 2]);
    }
  }
};

TEST_F(BubbleSortPerfTest, SeqPerformance) {
  BuzulukskyDBubbleSortSEQ task(input);

  auto start = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  const auto &result = task.GetOutput();
  EXPECT_TRUE(IsSorted(result));
  std::cout << "SEQ execution time: " << duration.count() << " ms\n";
}

TEST_F(BubbleSortPerfTest, MpiPerformance) {
  BuzulukskyDBubbleSortMPI task(input);

  auto start = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  const auto &result = task.GetOutput();
  EXPECT_TRUE(IsSorted(result));
  std::cout << "MPI execution time: " << duration.count() << " ms\n";
}

}  // namespace buzuluksky_d_bubble_sort
