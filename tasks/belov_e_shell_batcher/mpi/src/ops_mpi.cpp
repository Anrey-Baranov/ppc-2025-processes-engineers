#include "belov_e_shell_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "belov_e_shell_batcher/common/include/common.hpp"

namespace belov_e_shell_batcher {
BelovEShellBatcherMPI::BelovEShellBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BelovEShellBatcherMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool BelovEShellBatcherMPI::PreProcessingImpl() {
  return true;
}

void ShellSort(std::vector<int> &arr) {
  size_t n = arr.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = arr[i];
      size_t j = i;

      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }

      arr[j] = temp;
    }
  }
}

std::vector<int> BatcherLeftMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                  int local_arr_size, int rank, MPI_Comm comm) {
  std::vector<int> even_arr1;
  even_arr1.reserve(local_arr_size);
  std::vector<int> even_arr2;
  even_arr2.reserve(local_arr_size);
  for (int i = 0; i < local_arr_size; i += 2) {
    even_arr1.push_back(input_left_arr[i]);
  }
  for (int i = 0; i < local_arr_size; i += 2) {
    even_arr2.push_back(input_right_arr[i]);
  }
  std::vector<int> even_arr;
  even_arr.resize(even_arr1.size() + even_arr2.size());
  std::ranges::merge(even_arr1, even_arr2, even_arr.begin());

  std::vector<int> odd_arr;
  odd_arr.resize((2 * static_cast<size_t>(local_arr_size)) - even_arr.size());
  MPI_Sendrecv(even_arr.data(), static_cast<int>(even_arr.size()), MPI_INT, rank + 1, 0, odd_arr.data(),
               static_cast<int>(odd_arr.size()), MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);

  std::vector<int> left_arr;
  left_arr.reserve(local_arr_size);
  if (local_arr_size != 0) {
    left_arr.push_back(even_arr[0]);
  }
  int i = 0;
  while (left_arr.size() < static_cast<size_t>(local_arr_size)) {
    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    left_arr.push_back(comp.first);
    if (left_arr.size() < static_cast<size_t>(local_arr_size)) {
      left_arr.push_back(comp.second);
    }
    i++;
  }
  return left_arr;
}

std::vector<int> BatcherRightMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                   int local_arr_size, int rank, MPI_Comm comm) {
  std::vector<int> odd_arr1;
  odd_arr1.reserve(local_arr_size);
  std::vector<int> odd_arr2;
  odd_arr2.reserve(local_arr_size);
  for (int i = 1; i < local_arr_size; i += 2) {
    odd_arr1.push_back(input_left_arr[i]);
  }
  for (int i = 1; i < local_arr_size; i += 2) {
    odd_arr2.push_back(input_right_arr[i]);
  }

  std::vector<int> odd_arr;
  odd_arr.resize(odd_arr1.size() + odd_arr2.size());
  std::ranges::merge(odd_arr1, odd_arr2, odd_arr.begin());

  std::vector<int> even_arr;
  even_arr.resize((2 * static_cast<size_t>(local_arr_size)) - odd_arr.size());
  MPI_Sendrecv(odd_arr.data(), static_cast<int>(odd_arr.size()), MPI_INT, rank - 1, 0, even_arr.data(),
               static_cast<int>(even_arr.size()), MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);

  std::vector<int> right_arr;
  right_arr.reserve(local_arr_size);
  size_t i = 0;
  if (local_arr_size % 2 == 0) {
    i = (static_cast<size_t>(local_arr_size) / 2) - 1;

    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    right_arr.push_back(comp.second);
    i++;
  } else {
    i = (static_cast<size_t>(local_arr_size) - 1) / 2;
  }
  for (; i + 1 < even_arr.size() && i < odd_arr.size(); i++) {
    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    right_arr.push_back(comp.first);
    right_arr.push_back(comp.second);
  }
  for (; i + 1 < even_arr.size(); i++) {
    right_arr.push_back(even_arr[i + 1]);
  }
  for (; i < odd_arr.size(); i++) {
    right_arr.push_back(odd_arr[i]);
  }
  return right_arr;
}

void LeftProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  std::vector<int> right_arr;
  right_arr.resize(local_arr_size);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank + 1, 0, right_arr.data(), local_arr_size, MPI_INT,
               rank + 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = BatcherLeftMerge(local_arr, right_arr, local_arr_size, rank, comm);
}

void RightProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  std::vector<int> left_arr;
  left_arr.resize(local_arr_size);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank - 1, 0, left_arr.data(), local_arr_size, MPI_INT,
               rank - 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = BatcherRightMerge(left_arr, local_arr, local_arr_size, rank, comm);
}

void EvenPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  if (mpi_size % 2 != 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    LeftProcAct(rank, local_arr, local_arr_size, comm);
  } else {
    RightProcAct(rank, local_arr, local_arr_size, comm);
  }
}

void OddPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  if (rank == 0) {
    return;
  }
  if (mpi_size % 2 == 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    RightProcAct(rank, local_arr, local_arr_size, comm);
  } else {
    LeftProcAct(rank, local_arr, local_arr_size, comm);
  }
}

bool BelovEShellBatcherMPI::RunImpl() {
  int mpi_size = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> arr;
  int n = 0;
  if (rank == 0) {
    arr = GetInput();
    n = static_cast<int>(arr.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int garbage = (n % mpi_size == 0) ? 0 : (mpi_size - (n % mpi_size));

  if (rank == 0) {
    for (int i = 0; i < garbage; i++) {
      arr.push_back(std::numeric_limits<int>::max());
    }
  }

  int local_arr_size = (n + garbage) / mpi_size;
  std::vector<int> local_arr;
  local_arr.resize(local_arr_size);

  MPI_Scatter(arr.data(), local_arr_size, MPI_INT, local_arr.data(), local_arr_size, MPI_INT, 0, MPI_COMM_WORLD);

  ShellSort(local_arr);

  for (int i = 0; i < mpi_size + 1; i++) {
    if (i % 2 == 0) {
      EvenPhase(rank, mpi_size, local_arr, local_arr_size, MPI_COMM_WORLD);
    } else {
      OddPhase(rank, mpi_size, local_arr, local_arr_size, MPI_COMM_WORLD);
    }
  }

  arr.resize(n + garbage);
  MPI_Allgather(local_arr.data(), local_arr_size, MPI_INT, arr.data(), local_arr_size, MPI_INT, MPI_COMM_WORLD);
  arr.resize(n);
  GetOutput() = arr;
  return true;
}

bool BelovEShellBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace belov_e_shell_batcher
