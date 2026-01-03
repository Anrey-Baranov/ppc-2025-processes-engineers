#pragma once

#include <vector>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"
#include "task/include/task.hpp"

namespace alekseev_a_global_opt_chars {

class AlekseevAGlobalOptCharsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit AlekseevAGlobalOptCharsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SortingTrials();

  double ComputeLipschitzEstimate();
  [[nodiscard]] double ComputeIntervalCharacteristic(int interval_idx, double m_val) const;
  int BestInterval();

  double PerformTrial(double t);

  std::vector<TrialPoint> trial_points_;
  std::vector<double> t_points_;

  double lipschitz_estimate_{1.0};
  int peano_level_{10};
};

}  // namespace alekseev_a_global_opt_chars
