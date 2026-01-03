#include "kolotukhin_a_merge_sort_doubles/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kolotukhin_a_merge_sort_doubles/common/include/common.hpp"

namespace kolotukhin_a_merge_sort_doubles {

namespace {

std::vector<std::uint64_t> ConvertDoublesToKeys(const std::vector<double> &data) {
  const std::size_t data_size = data.size();
  std::vector<std::uint64_t> keys(data_size);

  for (std::size_t i = 0; i < data_size; i++) {
    std::uint64_t u = 0;
    std::memcpy(&u, &data[i], sizeof(double));
    if ((u & 0x8000000000000000ULL) != 0U) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    keys[i] = u;
  }

  return keys;
}

void ConvertKeysToDoubles(const std::vector<std::uint64_t> &keys, std::vector<double> &data) {
  const std::size_t data_size = data.size();
  for (std::size_t i = 0; i < data_size; ++i) {
    std::uint64_t u = keys[i];
    if ((u & 0x8000000000000000ULL) != 0U) {
      u &= ~0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    std::memcpy(&data[i], &u, sizeof(double));
  }
}

void RadixSortUint64(std::vector<std::uint64_t> &keys) {
  const std::size_t data_size = keys.size();
  if (data_size <= 1) {
    return;
  }

  const int radix_size = 256;
  std::vector<std::uint64_t> temp(data_size);

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<std::size_t> count(radix_size + 1, 0);

    for (std::size_t i = 0; i < data_size; i++) {
      const auto digit = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFF);
      ++count[digit + 1];
    }

    for (int i = 0; i < radix_size; ++i) {
      count[i + 1] += count[i];
    }

    for (std::size_t i = 0; i < data_size; ++i) {
      const auto digit = static_cast<std::uint8_t>((keys[i] >> shift) & 0xFF);
      const std::size_t pos = count[digit];
      temp[pos] = keys[i];
      count[digit] = pos + 1;
    }

    if (!temp.empty()) {
      for (std::size_t i = 0; i < data_size; ++i) {
        keys[i] = temp[i];
      }
    }
  }
}

void RadixSortDoubles(std::vector<double> &data) {
  const std::size_t data_size = data.size();
  if (data_size <= 1) {
    return;
  }

  std::vector<std::uint64_t> keys = ConvertDoublesToKeys(data);

  RadixSortUint64(keys);

  ConvertKeysToDoubles(keys, data);
}

std::vector<double> MergeSortedArrays(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> result;
  result.reserve(a.size() + b.size());
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) {
      result.push_back(a[i++]);
    } else {
      result.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    result.push_back(a[i++]);
  }
  while (j < b.size()) {
    result.push_back(b[j++]);
  }
  return result;
}

void HandleMergeAsReceiver(int rank, int step, int world_size, std::vector<double> &local_data) {
  int source_rank = rank + step;
  if (source_rank < world_size) {
    int remote_size = 0;
    MPI_Recv(&remote_size, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (remote_size > 0) {
      std::vector<double> remote_data(remote_size);
      MPI_Recv(remote_data.data(), remote_size, MPI_DOUBLE, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_data = MergeSortedArrays(local_data, remote_data);
    }
  }
}

void HandleMergeAsSender(int rank, int step, std::vector<double> &local_data) {
  int dest_rank = rank - step;
  int send_size = static_cast<int>(local_data.size());
  MPI_Send(&send_size, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
  if (send_size > 0) {
    MPI_Send(local_data.data(), send_size, MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
  }
  local_data.clear();
}

}  // namespace

KolotukhinAMergeSortDoublesMPI::KolotukhinAMergeSortDoublesMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = std::vector<double>();
  std::get<1>(GetOutput()) = -1;
}

bool KolotukhinAMergeSortDoublesMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  return (mpi_initialized != 0);
}

bool KolotukhinAMergeSortDoublesMPI::PreProcessingImpl() {
  std::get<0>(GetOutput()).clear();
  return true;
}

bool KolotukhinAMergeSortDoublesMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();

  int data_size = static_cast<int>(input.size());
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (data_size == 0) {
    std::get<0>(GetOutput()) = std::vector<double>();
    std::get<1>(GetOutput()) = rank;
    return true;
  }

  std::vector<int> displs(world_size, 0);
  std::vector<int> recv_counts(world_size, 0);
  int local_size = data_size / world_size;
  if (rank < data_size % world_size) {
    local_size++;
  }
  std::vector<double> local_data(local_size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      recv_counts[i] = (data_size / world_size) + (i < (data_size % world_size) ? 1 : 0);
      displs[i] = offset;
      offset += recv_counts[i];
    }
  }

  MPI_Bcast(recv_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? input.data() : nullptr, recv_counts.data(), displs.data(), MPI_DOUBLE, local_data.data(),
               local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  RadixSortDoubles(local_data);

  int step = 1;
  while (step < world_size) {
    if ((rank % (2 * step)) == 0) {
      HandleMergeAsReceiver(rank, step, world_size, local_data);
    } else if (((rank - step) % (2 * step)) == 0) {
      HandleMergeAsSender(rank, step, local_data);
    }
    step *= 2;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = local_data;
  std::get<1>(GetOutput()) = rank;

  return true;
}

bool KolotukhinAMergeSortDoublesMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_merge_sort_doubles
