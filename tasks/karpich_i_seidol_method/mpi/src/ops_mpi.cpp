#include "karpich_i_seidol_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "karpich_i_seidol_method/common/include/common.hpp"

namespace karpich_i_seidol_method {

KarpichISeidolMethodMPI::KarpichISeidolMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool KarpichISeidolMethodMPI::ValidationImpl() {
  return true;
}

bool KarpichISeidolMethodMPI::PreProcessingImpl() {
  return true;
}

bool KarpichISeidolMethodMPI::RunImpl() {
  double eps = std::get<3>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  double *a = nullptr;
  double *b = nullptr;

  std::size_t n = 0;
  if (rank == 0) {
    n = std::get<0>(GetInput());
    a = std::get<1>(GetInput()).data();
    b = std::get<2>(GetInput()).data();
  }
  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int step = static_cast<int>(n) / mpi_size;
  int remainder = static_cast<int>(n) % mpi_size;

  std::vector<int> send_counts_a(mpi_size, static_cast<int>(step * n));
  std::vector<int> send_counts_b(mpi_size, step);
  std::vector<int> displ_a(mpi_size, 0);
  std::vector<int> displ_b(mpi_size, 0);
  for (int i = 0; i < remainder; ++i) {
    send_counts_b[i]++;
    send_counts_a[i] += static_cast<int>(n);
  }
  for (int i = 1; i < mpi_size; ++i) {
    displ_b[i] = displ_b[i - 1] + send_counts_b[i - 1];
    displ_a[i] = displ_a[i - 1] + send_counts_a[i - 1];
  }
  std::vector<double> lb(send_counts_b[rank]);
  MPI_Scatterv(b, send_counts_b.data(), displ_b.data(), MPI_DOUBLE, lb.data(), static_cast<int>(lb.size()), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  std::vector<double> la(send_counts_a[rank]);
  MPI_Scatterv(a, send_counts_a.data(), displ_a.data(), MPI_DOUBLE, la.data(), static_cast<int>(la.size()), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(lb.size(), 0.0);
  std::vector<double> epsilons(n, 0.0);
  std::vector<double> epsilons_new(lb.size(), 0.0);
  bool iter_continue = true;
  while (iter_continue) {
    for (std::size_t i = 0; i < lb.size(); i++) {
      std::size_t row = displ_b[rank] + i;
      double ix = lb[i];
      for (std::size_t j = 0; j < row; j++) {
        ix = ix - (la[(i * n) + j] * x[j]);
      }
      for (std::size_t j = row + 1; j < n; j++) {
        ix = ix - (la[(i * n) + j] * x[j]);
      }

      ix = ix / la[(i * n) + row];
      epsilons[row] = std::fabs(ix - x[row]);
      epsilons_new[i] = epsilons[row];
      x[row] = ix;
      x_new[i] = ix;
    }
    MPI_Allgatherv(x_new.data(), static_cast<int>(x_new.size()), MPI_DOUBLE, x.data(), send_counts_b.data(),
                   displ_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(epsilons_new.data(), static_cast<int>(epsilons_new.size()), MPI_DOUBLE, epsilons.data(),
                   send_counts_b.data(), displ_b.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    iter_continue = IterContinue(epsilons, eps);
  }

  GetOutput() = x;
  if (rank != 0) {
    delete[] a;
    delete[] b;
  }
  return true;
}

bool KarpichISeidolMethodMPI::PostProcessingImpl() {
  return true;
}

bool KarpichISeidolMethodMPI::IterContinue(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *std::ranges::max_element(iter_eps);
  return max_in_iter > correct_eps;
}

}  // namespace karpich_i_seidol_method
