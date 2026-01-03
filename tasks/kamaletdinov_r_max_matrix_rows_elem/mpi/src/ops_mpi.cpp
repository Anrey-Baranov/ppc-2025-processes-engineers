#include "kamaletdinov_r_max_matrix_rows_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {

namespace {
void PrepareScattervArrays(int mpi_size, std::size_t total_size, std::vector<int> &sendcounts,
                           std::vector<int> &displs) {
  std::size_t procesess_step = total_size / mpi_size;
  std::size_t remainder = total_size % mpi_size;

  for (int i = 0; i < mpi_size; i++) {
    sendcounts[i] = static_cast<int>(procesess_step);
    if (std::cmp_less(i, remainder)) {
      sendcounts[i]++;
    }
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
  }
}

void ProcessLocalMatrix(const std::vector<int> &local_matrix, std::size_t start, std::size_t end, std::size_t m,
                        std::vector<int> &max_rows_elem) {
  std::size_t row = start / m;
  std::size_t local_idx = 0;

  if (!local_matrix.empty()) {
    max_rows_elem[row] = local_matrix[local_idx];
  }

  for (std::size_t i = start; i < end; i++) {
    if (i == ((row + 1) * m)) {
      row++;
      if (local_idx < local_matrix.size()) {
        max_rows_elem[row] = local_matrix[local_idx];
      }
    }
    if (local_idx < local_matrix.size()) {
      max_rows_elem[row] = std::max(max_rows_elem[row], local_matrix[local_idx]);
    }
    local_idx++;
  }
}

void MergeResults(int rank, int mpi_size, std::size_t n, std::vector<int> &max_rows_elem) {
  std::vector<int> gathered_data;
  std::vector<int> recvcounts;
  std::vector<int> displs;
  if (rank == 0) {
    gathered_data.resize(n * mpi_size);
    recvcounts.resize(mpi_size);
    displs.resize(mpi_size);
    for (int i = 0; i < mpi_size; i++) {
      recvcounts[i] = static_cast<int>(n);
      displs[i] = static_cast<int>(i * n);
    }
  }
  MPI_Gatherv(max_rows_elem.data(), static_cast<int>(n), MPI_INT, gathered_data.data(), recvcounts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (std::size_t i = 0; i < n; i++) {
      for (int j = 0; j < mpi_size; j++) {
        max_rows_elem[i] = std::max(max_rows_elem[i], gathered_data[(j * n) + i]);
      }
    }
  }
  MPI_Bcast(max_rows_elem.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
}
}  // namespace

KamaletdinovRMaxMatrixRowsElemMPI::KamaletdinovRMaxMatrixRowsElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KamaletdinovRMaxMatrixRowsElemMPI::ValidationImpl() {
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());
  valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
  return valid_;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PreProcessingImpl() {
  if (valid_) {
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &val = std::get<2>(GetInput());
    t_matrix_ = std::vector<int>(n * m);
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        t_matrix_[(j * m) + i] = val[(i * n) + j];
      }
    }
    return true;
  }
  return false;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::RunImpl() {
  // проверка корректности данных
  if (!valid_) {
    return false;
  }
  // получение размера матрицы
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());

  // debug
  //  std::string deb = "\n\n-----------\n";
  //  for(std::size_t i = 0; i < n; i++) {
  //    for(std::size_t j = 0; j < m; j++) {
  //      deb += std::to_string(t_matrix_[i*m + j]) + " ";
  //    }
  //    deb += "\n";
  //  }
  //  std::cout << deb;

  // данные о процессе
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Подготовка массивов для MPI_Scatterv
  std::vector<int> sendcounts(mpi_size);
  std::vector<int> displs(mpi_size);
  PrepareScattervArrays(mpi_size, t_matrix_.size(), sendcounts, displs);

  // Выделение памяти для локальной части матрицы
  std::vector<int> local_matrix(sendcounts[rank]);

  // Распределение матрицы с помощью MPI_Scatterv
  MPI_Scatterv(t_matrix_.data(), sendcounts.data(), displs.data(), MPI_INT, local_matrix.data(), sendcounts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  // Обработка локальной части матрицы
  std::size_t start = displs[rank];
  std::size_t end = start + sendcounts[rank];

  // выделение памяти для сохранения максимального элемента
  std::vector<int> max_rows_elem(n, 0);

  ProcessLocalMatrix(local_matrix, start, end, m, max_rows_elem);

  // Объединение результатов от всех процессов
  MergeResults(rank, mpi_size, n, max_rows_elem);

  // debug output
  //  std::cout << rank << ":";
  //  for(std::size_t i = 0; i < n; i++) {
  //    std::cout << max_rows_elem[i] << " ";
  //  }
  //  std::cout << std::endl;

  GetOutput() = max_rows_elem;

  return true;
}

bool KamaletdinovRMaxMatrixRowsElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
