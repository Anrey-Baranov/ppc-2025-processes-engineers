#pragma once

#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_m_jarvis_prohod {

class RomanovMJarvisProhodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RomanovMJarvisProhodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> JarvisMarch(std::vector<Point> points);
};

}  // namespace romanov_m_jarvis_prohod
