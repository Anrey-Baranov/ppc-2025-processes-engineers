#include "romanov_m_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "romanov_m_horizontal_matrix_vector/common/include/common.hpp"

namespace romanov_m_horizontal_matrix_vector {

RomanovMHorizontalMatrixVectorMPI::RomanovMHorizontalMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = std::vector<double>{};
}

bool RomanovMHorizontalMatrixVectorMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int check = 1;

  if (rank == 0) {
    const auto &input = GetInput();
    const auto &matrix = std::get<0>(input);
    const int rows = std::get<1>(input);
    const int cols = std::get<2>(input);
    const auto &vec = std::get<3>(input);

    if (rows <= 0 || cols <= 0) {
      check = 0;
    } else {
      bool mat_ok = (matrix.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols));
      bool vec_ok = (vec.size() == static_cast<size_t>(cols));
      if (!mat_ok || !vec_ok) {
        check = 0;
      }
    }
  }

  MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return check == 1;
}

bool RomanovMHorizontalMatrixVectorMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0;
  if (rank == 0) {
    rows = std::get<1>(GetInput());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows > 0) {
    GetOutput().resize(static_cast<size_t>(rows));
  }

  return true;
}

void RomanovMHorizontalMatrixVectorMPI::CalculateDistribution(int rows, int proc_num, std::vector<int> &counts,
                                                              std::vector<int> &displs) {
  int rows_per_proc = rows / proc_num;
  int remainder = rows % proc_num;

  counts.resize(static_cast<size_t>(proc_num));
  displs.resize(static_cast<size_t>(proc_num));

  int current_disp = 0;
  for (int i = 0; i < proc_num; ++i) {
    counts[static_cast<size_t>(i)] = rows_per_proc;
    if (i < remainder) {
      counts[static_cast<size_t>(i)]++;
    }
    displs[static_cast<size_t>(i)] = current_disp;
    current_disp += counts[static_cast<size_t>(i)];
  }
}

bool RomanovMHorizontalMatrixVectorMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  std::vector<double> vec;

  if (rank == 0) {
    rows = std::get<1>(GetInput());
    cols = std::get<2>(GetInput());
    vec = std::get<3>(GetInput());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    vec.resize(static_cast<size_t>(cols));
  }
  if (cols > 0) {
    MPI_Bcast(vec.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<int> rows_counts;
  std::vector<int> rows_displs;
  CalculateDistribution(rows, size, rows_counts, rows_displs);

  std::vector<int> send_counts(static_cast<size_t>(size));
  std::vector<int> send_displs(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    send_counts[static_cast<size_t>(i)] = rows_counts[static_cast<size_t>(i)] * cols;
    send_displs[static_cast<size_t>(i)] = rows_displs[static_cast<size_t>(i)] * cols;
  }

  int my_rows = rows_counts[static_cast<size_t>(rank)];
  int my_data_size = my_rows * cols;

  std::vector<double> local_matrix;
  if (my_data_size > 0) {
    local_matrix.resize(static_cast<size_t>(my_data_size));
  }

  const double *sendbuf = nullptr;
  if (rank == 0) {
    sendbuf = std::get<0>(GetInput()).data();
  }

  MPI_Scatterv(sendbuf, send_counts.data(), send_displs.data(), MPI_DOUBLE,
               (my_data_size > 0) ? local_matrix.data() : nullptr, my_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_res;
  if (my_rows > 0) {
    local_res.resize(static_cast<size_t>(my_rows));
  }

  for (int i = 0; i < my_rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      const std::size_t idx =
          (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j);
      sum += local_matrix[idx] * vec[static_cast<std::size_t>(j)];
    }
    local_res[static_cast<size_t>(i)] = sum;
  }

  MPI_Allgatherv((my_rows > 0) ? local_res.data() : nullptr, my_rows, MPI_DOUBLE, GetOutput().data(),
                 rows_counts.data(), rows_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  return true;
}

bool RomanovMHorizontalMatrixVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_horizontal_matrix_vector
