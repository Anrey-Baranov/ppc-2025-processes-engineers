#pragma once

#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_m_jarvis_prohod {

class RomanovMJarvisProhodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() noexcept {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RomanovMJarvisProhodMPI(const InType &in);

  static std::vector<Point> JarvisMarch(std::vector<Point> points);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> FinalHull(int rank, std::vector<Point> &all_hull_points);
};

}  // namespace romanov_m_jarvis_prohod
