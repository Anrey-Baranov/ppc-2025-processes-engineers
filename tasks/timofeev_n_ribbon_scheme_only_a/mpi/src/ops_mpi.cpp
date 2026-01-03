#include "timofeev_n_ribbon_scheme_only_a/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "timofeev_n_ribbon_scheme_only_a/common/include/common.hpp"

namespace timofeev_n_ribbon_scheme_only_a {

TimofeevNRibbonSchemeOnlyAMPI::TimofeevNRibbonSchemeOnlyAMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<std::vector<int>>(0, std::vector<int>(0));
}

bool TimofeevNRibbonSchemeOnlyAMPI::ValidationImpl() {
  std::vector<std::vector<int>> &a = GetInput().first;
  std::vector<std::vector<int>> &b = GetInput().second;
  size_t n = a.size();
  size_t m2 = a[0].size();
  size_t m1 = b.size();
  size_t k = b[0].size();
  return (m2 == m1 && n != 0 && m1 != 0 && m2 != 0 && k != 0);
}

bool TimofeevNRibbonSchemeOnlyAMPI::PreProcessingImpl() {
  return true;
}

void TimofeevNRibbonSchemeOnlyAMPI::SendingAParts(MatrixType &a, int &size, size_t k) {
  if (a.empty()) {
    return;
  }
  size_t a_row_size = a[0].size();
  size_t a_size = a.size();
  size_t i = 0;
  size_t iter = 0;
  std::vector<size_t> sizes(size - 1, 0);
  for (size_t p_ind = 0; p_ind < sizes.size() - 1; p_ind++) {
    sizes[p_ind] = (a_size >= k ? k : 0);
    if (a_size >= k) {
      a_size -= k;
      iter++;
    }
  }
  sizes[size - 2] = a_size;
  iter = 0;
  for (; i < sizes.size(); i++) {
    MPI_Send(&sizes[i], 1, MPI_UNSIGNED_LONG, static_cast<int>(i + 1), 0, MPI_COMM_WORLD);
    for (size_t j = 0; j < sizes[i]; j++) {
      MPI_Send(a[iter].data(), static_cast<int>(a_row_size), MPI_INT, static_cast<int>(i + 1), 0, MPI_COMM_WORLD);
      iter++;
    }
  }
}

void TimofeevNRibbonSchemeOnlyAMPI::ReceivingCParts(MatrixType &cmatr, int &size, size_t k, size_t &b_row_size) {
  if (cmatr.empty()) {
    return;
  }
  size_t c_size = cmatr.size();
  size_t i = 0;
  size_t iter = 0;
  std::vector<size_t> sizes(size - 1, 0);
  for (size_t p_ind = 0; p_ind < sizes.size() - 1; p_ind++) {
    sizes[p_ind] = (c_size >= k ? k : 0);
    if (c_size >= k) {
      c_size -= k;
      iter++;
    }
  }
  sizes[size - 2] = c_size;
  iter = 0;
  for (; i < sizes.size(); i++) {
    for (size_t j = 0; j < sizes[i]; j++) {
      MPI_Recv(cmatr[iter].data(), static_cast<int>(b_row_size), MPI_INT, static_cast<int>(i + 1), 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      iter++;
    }
  }
}

int TimofeevNRibbonSchemeOnlyAMPI::CalculatingCElement(MatrixType &a_part, MatrixType &b_copy, size_t &i, size_t &j) {
  int summ = 0;
  for (size_t kk = 0; kk < a_part[0].size(); kk++) {
    summ += a_part[i][kk] * b_copy[kk][j];
  }
  return summ;
}

void TimofeevNRibbonSchemeOnlyAMPI::CalculatingCPart(size_t &k, MatrixType &a_part, MatrixType &b_copy,
                                                     MatrixType &c_part) {
  for (size_t i = 0; i < k; i++) {
    for (size_t j = 0; j < b_copy[0].size(); j++) {
      c_part[i][j] = CalculatingCElement(a_part, b_copy, i, j);
    }
  }
}

void TimofeevNRibbonSchemeOnlyAMPI::BroadcastingParameters(size_t &a_size, size_t &a_row_size, size_t &b_size,
                                                           size_t &b_row_size) {
  MPI_Bcast(&a_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_row_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_row_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
}

void TimofeevNRibbonSchemeOnlyAMPI::BroadcastingB(MatrixType &b) {
  if (b.empty() || b[0].empty()) {
    return;
  }
  size_t b_row_size = b[0].size();
  for (auto &i : b) {
    MPI_Bcast(i.data(), static_cast<int>(b_row_size), MPI_INT, 0, MPI_COMM_WORLD);
  }
}

int TimofeevNRibbonSchemeOnlyAMPI::CalculatingCElementOneProcess(MatrixType &a, MatrixType &b, size_t &i, size_t &j) {
  int summ = 0;
  for (size_t kk = 0; kk < b.size(); kk++) {
    summ += a[i][kk] * b[kk][j];
  }
  return summ;
}

void TimofeevNRibbonSchemeOnlyAMPI::CalculatingCOneProcess(MatrixType &a, MatrixType &b, MatrixType &c) {
  for (size_t i = 0; i < a.size(); i++) {
    c[i].resize(b[0].size());
    for (size_t j = 0; j < b[0].size(); j++) {
      c[i][j] = CalculatingCElementOneProcess(a, b, i, j);
    }
  }
}

bool TimofeevNRibbonSchemeOnlyAMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    MatrixType a = GetInput().first;
    MatrixType b = GetInput().second;
    GetOutput().resize(a.size());
    CalculatingCOneProcess(a, b, GetOutput());
    return true;
  }

  if (rank == 0) {
    MatrixType a = GetInput().first;
    MatrixType b = GetInput().second;
    size_t k = 0;
    k = (a.size() / static_cast<size_t>(size - 1)) + (a.size() % static_cast<size_t>(size - 1) > 0 ? 1 : 0);
    size_t a_size = a.size();
    size_t a_row_size = a[0].size();
    size_t b_size = b.size();
    size_t b_row_size = b[0].size();
    BroadcastingParameters(a_size, a_row_size, b_size, b_row_size);
    BroadcastingB(b);
    SendingAParts(a, size, k);
    std::vector<std::vector<int>> cmatr(a.size(), std::vector<int>(b_row_size));
    ReceivingCParts(cmatr, size, k, b_row_size);
    for (auto &i : cmatr) {  // рассылка того, что получилось
      MPI_Bcast(i.data(), static_cast<int>(b_row_size), MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = cmatr;
  } else {
    size_t k = 0;
    size_t a_size = 0;
    size_t a_row_size = 0;
    size_t b_size = 0;
    size_t b_row_size = 0;
    BroadcastingParameters(a_size, a_row_size, b_size, b_row_size);
    std::vector<std::vector<int>> b_copy(b_size, std::vector<int>(b_row_size));
    BroadcastingB(b_copy);
    MPI_Recv(&k, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<std::vector<int>> a_part(k, std::vector<int>(a_row_size));
    for (size_t j = 0; j < k; j++) {  // принимаем строки A
      MPI_Recv(a_part[j].data(), static_cast<int>(a_row_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::vector<std::vector<int>> c_part(k, std::vector<int>(b_row_size));
    CalculatingCPart(k, a_part, b_copy, c_part);  // можем вычислить k строк (k * b_row_size элементов) матрицы C
    for (size_t i = 0; i < k; i++) {              // у нас есть матрица k * b_row_size
      MPI_Send(c_part[i].data(), static_cast<int>(b_row_size), MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    GetOutput().resize(a_size);
    for (size_t i = 0; i < a_size; i++) {  // принимает результат из Bcast
      GetOutput()[i].resize(b_row_size);
      MPI_Bcast(GetOutput()[i].data(), static_cast<int>(b_row_size), MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool TimofeevNRibbonSchemeOnlyAMPI::PostProcessingImpl() {
  return true;
}

}  // namespace timofeev_n_ribbon_scheme_only_a
