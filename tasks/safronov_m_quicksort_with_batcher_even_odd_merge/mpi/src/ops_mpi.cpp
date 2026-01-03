#include "safronov_m_quicksort_with_batcher_even_odd_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "safronov_m_quicksort_with_batcher_even_odd_merge/common/include/common.hpp"

namespace safronov_m_quicksort_with_batcher_even_odd_merge {

SafronovMQuicksortWithBatcherEvenOddMergeMPI::SafronovMQuicksortWithBatcherEvenOddMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SafronovMQuicksortWithBatcherEvenOddMergeMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool SafronovMQuicksortWithBatcherEvenOddMergeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::SendingVector(int rank) {
  int size_vec = 0;
  if (rank == 0) {
    size_vec = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&size_vec, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetInput().resize(size_vec);
  }
  MPI_Bcast(GetInput().data(), size_vec, MPI_INT, 0, MPI_COMM_WORLD);
}

int SafronovMQuicksortWithBatcherEvenOddMergeMPI::LengthsLocalArrays(int size_arr, int rank, int size) {
  int local_size = size_arr / (size);
  if ((rank) < (size_arr % size)) {
    local_size += 1;
  }
  return local_size;
}

std::vector<int> SafronovMQuicksortWithBatcherEvenOddMergeMPI::CalculatingInterval(int size_prcs, int rank,
                                                                                   int size_arr) {
  std::vector<int> vec(2);
  int whole_part = size_arr / size_prcs;
  int real_part = size_arr % size_prcs;
  int start = rank * whole_part;
  if ((rank - 1 < real_part) && (rank - 1 != -1)) {
    start += rank;
  } else if (rank != 0) {
    start += real_part;
  }
  int end = start + whole_part - 1;
  if (rank < real_part) {
    end += 1;
  }
  vec[0] = start;
  vec[1] = end;
  return vec;
}

std::pair<int, int> SafronovMQuicksortWithBatcherEvenOddMergeMPI::SplitRange(std::vector<int> &array, int left,
                                                                             int right) {
  int i = left;
  int j = right;
  int mid = left + ((right - left) / 2);
  int pivot = array[mid];

  while (i <= j) {
    while (array[i] < pivot) {
      i++;
    }
    while (array[j] > pivot) {
      j--;
    }
    if (i <= j) {
      int tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
      i++;
      j--;
    }
  }

  return {i, j};
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::QuickSort(std::vector<int> &array) {
  if (array.empty()) {
    return;
  }
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(0, static_cast<int>(array.size()) - 1);

  while (!stack.empty()) {
    auto range = stack.back();
    stack.pop_back();
    int left = range.first;
    int right = range.second;
    if (left >= right) {
      continue;
    }
    auto borders = SplitRange(array, left, right);
    if (left < borders.second) {
      stack.emplace_back(left, borders.second);
    }
    if (borders.first < right) {
      stack.emplace_back(borders.first, right);
    }
  }
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::MergeAndSplit(std::vector<int> &own_data,
                                                                 std::vector<int> &neighbor_data, bool flag) {
  std::vector<int> data(own_data.size() + neighbor_data.size());
  // std::merge(own_data.begin(), own_data.end(), neighbor_data.begin(), neighbor_data.end(), data.begin());
  std::ranges::merge(own_data, neighbor_data, data.begin());
  if (!flag) {
    auto mid = data.begin() + static_cast<std::ptrdiff_t>(own_data.size());
    std::copy(data.begin(), mid, own_data.begin());
  } else {
    auto start = data.begin() + static_cast<std::ptrdiff_t>(neighbor_data.size());
    std::copy(start, data.end(), own_data.begin());
  }
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::DataExchange(std::vector<int> &own_data, int size_arr, int rank,
                                                                int size, int neighbor) {
  MPI_Status status;
  int neighbor_size = LengthsLocalArrays(size_arr, rank + neighbor, size);
  std::vector<int> neighbor_data(neighbor_size);
  MPI_Sendrecv(own_data.data(), static_cast<int>(own_data.size()), MPI_INT, rank + neighbor, 0, neighbor_data.data(),
               neighbor_size, MPI_INT, rank + neighbor, 0, MPI_COMM_WORLD, &status);
  bool flag = (neighbor != 1);
  MergeAndSplit(own_data, neighbor_data, flag);
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::EvenPhase(std::vector<int> &own_data, std::vector<int> &interval,
                                                             int size_arr, int rank, int size) {
  if ((rank % 2 == 0) && (rank + 1 < size) && (interval[1] != size_arr - 1) && (interval[0] <= interval[1])) {
    DataExchange(own_data, size_arr, rank, size, 1);
  } else if ((interval[0] <= interval[1]) && (rank % 2 == 1)) {
    DataExchange(own_data, size_arr, rank, size, -1);
  }
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::OddPhase(std::vector<int> &own_data, std::vector<int> &interval,
                                                            int size_arr, int rank, int size) {
  if ((rank % 2 == 1) && (rank + 1 < size) && (interval[1] != size_arr - 1) && (interval[0] <= interval[1])) {
    DataExchange(own_data, size_arr, rank, size, 1);
  } else if ((interval[0] <= interval[1]) && (rank != 0) && (rank % 2 == 0)) {
    DataExchange(own_data, size_arr, rank, size, -1);
  }
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::BatcherOddEvenPhases(std::vector<int> &own_data,
                                                                        std::vector<int> &interval, int size_arr,
                                                                        int rank, int size) {
  int min_local_block_size = size_arr / size;
  if (min_local_block_size == 0) {
    min_local_block_size = 1;
  }
  int phases = (size_arr + min_local_block_size - 1) / min_local_block_size;
  for (int i = 0; i < phases; i++) {
    if (i % 2 == 0) {
      EvenPhase(own_data, interval, size_arr, rank, size);
    } else {
      OddPhase(own_data, interval, size_arr, rank, size);
    }
  }
}

void SafronovMQuicksortWithBatcherEvenOddMergeMPI::SendingResult(int rank) {
  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(total_size);
  }

  MPI_Bcast(GetOutput().data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);
}

bool SafronovMQuicksortWithBatcherEvenOddMergeMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  SendingVector(rank);

  if (rank == 0) {
    int size_arr = static_cast<int>(GetInput().size());
    std::vector<int> sizes_local_arrays(size);
    for (int i = 1; i < size; i++) {
      std::vector<int> interval = CalculatingInterval(size, i, size_arr);
      sizes_local_arrays[i] = (interval[1] + 1) - interval[0];
      MPI_Send(interval.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    std::vector<int> interval = CalculatingInterval(size, 0, size_arr);
    std::vector<int> own_data = {};
    if (interval[0] <= interval[1]) {
      own_data = std::vector<int>(GetInput().begin() + interval[0], GetInput().begin() + interval[1] + 1);
      QuickSort(own_data);
    }
    BatcherOddEvenPhases(own_data, interval, size_arr, rank, size);
    GetOutput().insert(GetOutput().end(), own_data.begin(), own_data.end());
    MPI_Status status;
    for (int i = 1; i < size; i++) {
      std::vector<int> buf(sizes_local_arrays[i]);
      MPI_Recv(buf.data(), sizes_local_arrays[i], MPI_INT, i, 2, MPI_COMM_WORLD, &status);
      GetOutput().insert(GetOutput().end(), buf.begin(), buf.end());
    }
  } else {
    MPI_Status status;
    std::vector<int> buf(2);
    MPI_Recv(buf.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    std::vector<int> own_data = {};
    if (buf[0] <= buf[1]) {
      own_data = std::vector<int>(GetInput().begin() + buf[0], GetInput().begin() + buf[1] + 1);
      QuickSort(own_data);
    }
    int size_arr = static_cast<int>(GetInput().size());
    BatcherOddEvenPhases(own_data, buf, size_arr, rank, size);
    MPI_Send(own_data.data(), static_cast<int>(own_data.size()), MPI_INT, 0, 2, MPI_COMM_WORLD);
  }
  SendingResult(rank);
  return true;
}

bool SafronovMQuicksortWithBatcherEvenOddMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace safronov_m_quicksort_with_batcher_even_odd_merge
