#pragma once

#include <cstddef>

#include "task/include/task.hpp"
#include "timofeev_n_ribbon_scheme_only_a/common/include/common.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

class TimofeevNRibbonSchemeOnlyAMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TimofeevNRibbonSchemeOnlyAMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void SendingAParts(MatrixType &a, int &size, size_t k);
  static void ReceivingCParts(MatrixType &cmatr, int &size, size_t k, size_t &b_row_size);
  static int CalculatingCElement(MatrixType &a_part, MatrixType &b_copy, size_t &i, size_t &j);
  static void CalculatingCPart(size_t &k, MatrixType &a_part, MatrixType &b_copy, MatrixType &c_part);
  static void BroadcastingParameters(size_t &a_size, size_t &a_row_size, size_t &b_size, size_t &b_row_size);
  static void BroadcastingB(MatrixType &b);
  static int CalculatingCElementOneProcess(MatrixType &a, MatrixType &b, size_t &i, size_t &j);
  static void CalculatingCOneProcess(MatrixType &a, MatrixType &b, MatrixType &c);
};

}  // namespace timofeev_n_ribbon_scheme_only_a
