#include "karpich_i_integrals_multistep_rectangle/seq/include/ops_seq.hpp"

#include <cmath>

#include "karpich_i_integrals_multistep_rectangle/common/include/common.hpp"
#include "task/include/task.hpp"

namespace karpich_i_integrals_multistep_rectangle {

namespace {

double IntegrandFunction(double x, double y, double z) {
  return (x * y * z) + (x * x) + (y * y) + (z * z);
}

double ComputeRectangleIntegral(int n, int step) {
  double result = 0.0;
  int divisions = n * step;
  double h = 1.0 / divisions;

  for (int i = 0; i < divisions; i++) {
    for (int j = 0; j < divisions; j++) {
      for (int k = 0; k < divisions; k++) {
        double x = (i + 0.5) * h;
        double y = (j + 0.5) * h;
        double z = (k + 0.5) * h;
        result += IntegrandFunction(x, y, z) * h * h * h;
      }
    }
  }

  return result;
}

}  // namespace

KarpichIIntegralsMultistepRectangleSEQ::KarpichIIntegralsMultistepRectangleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIIntegralsMultistepRectangleSEQ::ValidationImpl() {
  if (GetInput() <= 0) {
    return false;
  }
  if (GetOutput() != 0) {
    return false;
  }
  if (GetInput() > 1000) {
    return false;
  }
  if (GetStaticTypeOfTask() != ppc::task::TypeOfTask::kSEQ) {
    return false;
  }
  return true;
}

bool KarpichIIntegralsMultistepRectangleSEQ::PreProcessingImpl() {
  if (GetInput() <= 0) {
    return false;
  }
  GetOutput() = 0;
  if (GetOutput() != 0) {
    return false;
  }
  int divisions = GetInput() * 3;
  return divisions > 0;
}

bool KarpichIIntegralsMultistepRectangleSEQ::RunImpl() {
  if (GetInput() == 0) {
    return false;
  }

  double step1_result = ComputeRectangleIntegral(GetInput(), 1);
  double step2_result = ComputeRectangleIntegral(GetInput(), 2);
  double step3_result = ComputeRectangleIntegral(GetInput(), 3);

  double final_result = (step1_result + step2_result + step3_result) / 3.0;

  GetOutput() = static_cast<OutType>(std::round(final_result));
  return GetOutput() > 0;
}

bool KarpichIIntegralsMultistepRectangleSEQ::PostProcessingImpl() {
  if (GetOutput() <= 0) {
    return false;
  }
  if (GetInput() <= 0) {
    return false;
  }
  auto min_expected = static_cast<OutType>(1);
  if (GetOutput() < min_expected) {
    return false;
  }
  auto max_expected = static_cast<OutType>(GetInput() * GetInput() * GetInput() * 10);
  if (GetOutput() > max_expected) {
    return false;
  }
  if (GetOutput() == GetInput() && GetInput() > 1) {
    return false;
  }
  return true;
}

}  // namespace karpich_i_integrals_multistep_rectangle
