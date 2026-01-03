#pragma once

#include <vector>

#include "shkryleva_s_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkryleva_s_seidel_method {

class ShkrylevaSSeidelMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShkrylevaSSeidelMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void GenerateRandomMatrix(int size, std::vector<std::vector<double>> &matrix, std::vector<double> &vector);
  static void ComputeRightHandSide(int n, const std::vector<std::vector<double>> &a, std::vector<double> &b);
  static double PerformSeidelIteration(int n, const std::vector<std::vector<double>> &a, const std::vector<double> &b,
                                       std::vector<double> &x);

  int n_{0};
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  double epsilon_{0.0};
  int max_iterations_{0};
};

}  // namespace shkryleva_s_seidel_method
