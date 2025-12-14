#pragma once

#include <mpi.h>

#include <iostream>

#include "baranov_a_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baranov_a_dijkstra_crs {

class BaranovADijkstraCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  // Добавляем статический метод проверки
  static bool CanCreate() {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    return mpi_initialized != 0;
  }

  explicit BaranovADijkstraCrsMPI(const InType &in);

  // Убираем деструктор, пусть компилятор сгенерирует его
  ~BaranovADijkstraCrsMPI() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  template <typename T>
  std::vector<T> DijkstraParallelTemplate(int vertices, const std::vector<int> &row_ptr,
                                          const std::vector<int> &col_idx, const std::vector<T> &values, int source);
};

}  // namespace baranov_a_dijkstra_crs