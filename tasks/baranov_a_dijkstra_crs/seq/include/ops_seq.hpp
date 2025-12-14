#pragma once

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BaranovADijkstraCrsSEQ(const InType &in);

  // Убираем деструктор
  ~BaranovADijkstraCrsSEQ() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  template <typename T>
  std::vector<T> DijkstraSequentialTemplate(int vertices, const std::vector<int> &row_ptr,
                                            const std::vector<int> &col_idx, const std::vector<T> &values, int source);
};

}  // namespace baranov_a_dijkstra_crs