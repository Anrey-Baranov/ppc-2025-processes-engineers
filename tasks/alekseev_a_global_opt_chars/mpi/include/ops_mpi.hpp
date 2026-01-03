#pragma once

#include <vector>

#include "alekseev_a_global_opt_chars/common/include/common.hpp"
#include "task/include/task.hpp"

namespace alekseev_a_global_opt_chars {

class AlekseevAGlobalOptCharsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit AlekseevAGlobalOptCharsMPI(const InType &in);

 private:
  std::vector<TrialPoint> trial_points_;
  std::vector<double> t_points_;

  double lipschitz_estimate_{1.0};
  int peano_level_{10};

  int world_rank_{0};
  int world_size_{1};

  void SortingTrials();
  double ComputeLipschitzEstimate();

  void CharacteristicsParallelComputed(double m_val, std::vector<double> &characteristics);

  static int BestInterval(const std::vector<double> &characteristics);

  double PerformTrial(double t);

  void BroadcastTrialData();

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace alekseev_a_global_opt_chars
