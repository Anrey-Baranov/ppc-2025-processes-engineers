#pragma once

#include <vector>

#include "kolotukhin_a_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kolotukhin_a_hypercube {

class KolotukhinAHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KolotukhinAHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CalculateHypercubeDimension(int num_proc);
  static int GetNeighbor(int rank, int dim);
  static void PerformComputeLoad(int iterations);
  static void SendData(std::vector<int> &data, int next_neighbor);
  static void RecvData(std::vector<int> &data, int prev_neighbor);
  static void CalcPositions(int my_rank, std::vector<int> &path, int &my_pos, int &next, int &prev);
  static std::vector<int> CalcPath(int source, int dest, int dimensions);
  static std::vector<int> CalcPathLowToHigh(int source, int dest, int dimensions, int xor_val);
  static std::vector<int> CalcPathHighToLow(int source, int dest, int dimensions, int xor_val);
};

}  // namespace kolotukhin_a_hypercube
