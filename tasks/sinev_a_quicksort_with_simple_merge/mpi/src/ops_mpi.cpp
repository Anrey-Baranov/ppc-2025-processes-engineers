#include "sinev_a_quicksort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

SinevAQuicksortWithSimpleMergeMPI::SinevAQuicksortWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool SinevAQuicksortWithSimpleMergeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (GetInput().empty()) {
      return false;
    }
  }

  return true;
}

bool SinevAQuicksortWithSimpleMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
  }
  return true;
}

int SinevAQuicksortWithSimpleMergeMPI::Partition(std::vector<int> &arr, int left, int right) {
  int pivot_index = left + ((right - left) / 2);
  int pivot_value = arr[pivot_index];

  std::swap(arr[pivot_index], arr[left]);

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) {
      i++;
    }

    while (i <= j && arr[j] > pivot_value) {
      j--;
    }

    if (i < j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    } else {
      break;
    }
  }

  std::swap(arr[left], arr[j]);
  return j;
}

void SinevAQuicksortWithSimpleMergeMPI::SimpleMerge(std::vector<int> &arr, int left, int mid, int right) {
  if (left >= right) {
    return;
  }

  std::vector<int> temp(right - left + 1);

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }

  while (j <= right) {
    temp[k++] = arr[j++];
  }

  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void SinevAQuicksortWithSimpleMergeMPI::QuickSortWithSimpleMerge(std::vector<int> &arr, int left, int right) {
  struct Range {
    int left;
    int right;
  };

  std::stack<Range> stack;
  stack.push({left, right});

  while (!stack.empty()) {
    Range current = stack.top();
    stack.pop();

    int l = current.left;
    int r = current.right;

    if (l >= r) {
      continue;
    }

    int pivot_index = Partition(arr, l, r);

    int left_size = pivot_index - l;
    int right_size = r - pivot_index;

    if (left_size > 1 && right_size > 1) {
      if (left_size > right_size) {
        stack.push({pivot_index + 1, r});
        stack.push({l, pivot_index - 1});
      } else {
        stack.push({l, pivot_index - 1});
        stack.push({pivot_index + 1, r});
      }
    } else if (left_size > 1) {
      stack.push({l, pivot_index - 1});
    } else if (right_size > 1) {
      stack.push({pivot_index + 1, r});
    }

    SimpleMerge(arr, l, pivot_index, r);
  }
}

SinevAQuicksortWithSimpleMergeMPI::DistributionInfo SinevAQuicksortWithSimpleMergeMPI::PrepareDistributionInfo(
    int total_size, int world_size, int world_rank) {
  DistributionInfo info;

  int base_size = total_size / world_size;
  int remainder = total_size % world_size;

  info.local_size = base_size + (world_rank < remainder ? 1 : 0);

  info.send_counts.assign(world_size, 0);
  info.displacements.assign(world_size, 0);

  if (world_rank == 0) {
    int displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      info.send_counts[i] = base_size + (i < remainder ? 1 : 0);
      info.displacements[i] = displacement;
      displacement += info.send_counts[i];
    }
  }

  return info;
}

std::vector<int> SinevAQuicksortWithSimpleMergeMPI::DistributeData(int world_size, int world_rank) {
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  DistributionInfo info = PrepareDistributionInfo(total_size, world_size, world_rank);

  std::vector<int> local_buffer(info.local_size);

  MPI_Scatterv(GetOutput().data(), info.send_counts.data(), info.displacements.data(), MPI_INT, local_buffer.data(),
               info.local_size, MPI_INT, 0, MPI_COMM_WORLD);

  return local_buffer;
}

void SinevAQuicksortWithSimpleMergeMPI::PerformMultiWayMerge(const std::vector<int> &all_sizes) {
  std::vector<std::pair<int, int>> segments;
  int offset = 0;

  for (int all_size : all_sizes) {
    if (all_size > 0) {
      segments.emplace_back(offset, offset + all_size - 1);
      offset += all_size;
    }
  }

  while (segments.size() > 1) {
    std::vector<std::pair<int, int>> next_segments;

    for (std::size_t i = 0; i < segments.size(); i += 2) {
      if (i + 1 < segments.size()) {
        int left = segments[i].first;
        int mid = segments[i].second;
        int right = segments[i + 1].second;
        SimpleMerge(GetOutput(), left, mid, right);
        next_segments.emplace_back(left, right);
      } else {
        next_segments.push_back(segments[i]);
      }
    }

    segments = std::move(next_segments);
  }
}

void SinevAQuicksortWithSimpleMergeMPI::BroadcastFinalResult() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int final_size = 0;
  if (world_rank == 0) {
    final_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank != 0) {
    GetOutput().resize(final_size);
  }

  MPI_Bcast(GetOutput().data(), final_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void SinevAQuicksortWithSimpleMergeMPI::ParallelQuickSort() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size == 1) {
    if (!GetOutput().empty()) {
      QuickSortWithSimpleMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
    }
    return;
  }

  // 1. Распределение данных по процессам
  std::vector<int> local_buffer = DistributeData(world_size, world_rank);

  // 2. Локальная сортировка на каждом процессе
  if (!local_buffer.empty()) {
    QuickSortWithSimpleMerge(local_buffer, 0, static_cast<int>(local_buffer.size()) - 1);
  }

  // 3. Сбор размеров от всех процессов
  std::vector<int> all_sizes(world_size);
  int local_size = static_cast<int>(local_buffer.size());

  MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 4. Подготовка буфера на корневом процессе
  std::vector<int> displacements(world_size, 0);
  int total_size = 0;

  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      displacements[i] = total_size;
      total_size += all_sizes[i];
    }
    GetOutput().resize(total_size);
  }

  // 5. Сбор данных на корневом процессе
  int *recv_data = (world_rank == 0) ? GetOutput().data() : nullptr;
  int *recv_counts = (world_rank == 0) ? all_sizes.data() : nullptr;
  int *recv_displs = (world_rank == 0) ? displacements.data() : nullptr;

  MPI_Gatherv(local_buffer.data(), local_size, MPI_INT, recv_data, recv_counts, recv_displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  // 6. Многостороннее слияние отсортированных частей на процессе 0
  if (world_rank == 0 && !GetOutput().empty()) {
    PerformMultiWayMerge(all_sizes);
  }

  // 7. Рассылка результата (оставляем для корректной работы тестов)
  BroadcastFinalResult();
}

bool SinevAQuicksortWithSimpleMergeMPI::RunImpl() {
  ParallelQuickSort();
  return true;
}

bool SinevAQuicksortWithSimpleMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_quicksort_with_simple_merge
