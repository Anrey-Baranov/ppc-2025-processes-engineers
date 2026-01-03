#include "shkryleva_s_vec_min_val/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <vector>

#include "shkryleva_s_vec_min_val/common/include/common.hpp"

namespace shkryleva_s_vec_min_val {

namespace {

struct DistributionInfo {
  std::vector<int> sendcounts;
  std::vector<int> displacements;
  int local_count = 0;
};

DistributionInfo CalculateDistribution(int total_size, int world_size) {
  DistributionInfo info;
  info.sendcounts.resize(world_size);
  info.displacements.resize(world_size);

  const int base_size = total_size / world_size;
  const int extra_items = total_size % world_size;

  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    info.sendcounts[i] = base_size + (i < extra_items ? 1 : 0);
    info.displacements[i] = offset;
    offset += info.sendcounts[i];
  }

  return info;
}

void BroadcastVectorSize(uint64_t &total_size_uint64) {
  MPI_Bcast(&total_size_uint64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
}

int ComputeLocalMinimum(const std::vector<int> &local_data) {
  if (local_data.empty()) {
    return INT_MAX;
  }

  int local_min = INT_MAX;
  for (int value : local_data) {
    local_min = std::min(value, local_min);
  }
  return local_min;
}

std::vector<int> ScatterVectorData(const std::vector<int> *input_data_ptr, const DistributionInfo &info,
                                   int world_rank) {
  std::vector<int> local_data;

  if (info.local_count > 0) {
    local_data.resize(info.local_count);

    MPI_Scatterv((world_rank == 0) ? input_data_ptr->data() : nullptr, info.sendcounts.data(),
                 info.displacements.data(), MPI_INT, local_data.data(), info.local_count, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return local_data;
}

int PerformGlobalReduction(int local_min) {
  int total_min = INT_MAX;
  MPI_Allreduce(&local_min, &total_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return total_min;
}

}  // namespace

ShkrylevaSVecMinValMPI::ShkrylevaSVecMinValMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrylevaSVecMinValMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  bool is_valid = true;

  if (world_rank == 0) {
    if (!GetInput().empty()) {
      auto size = static_cast<uint64_t>(GetInput().size());
      is_valid = (size <= static_cast<uint64_t>(std::numeric_limits<int>::max()));
    }
  }

  int validation_result = is_valid ? 1 : 0;
  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation_result != 0;
}

bool ShkrylevaSVecMinValMPI::PreProcessingImpl() {
  GetOutput() = INT_MAX;
  return true;
}

bool ShkrylevaSVecMinValMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  uint64_t total_size_uint64 = 0;
  const std::vector<int> *input_data_ptr = nullptr;

  if (world_rank == 0) {
    input_data_ptr = &GetInput();
    total_size_uint64 = static_cast<uint64_t>(input_data_ptr->size());
  }

  BroadcastVectorSize(total_size_uint64);

  const int total_size = static_cast<int>(total_size_uint64);
  int local_min = INT_MAX;

  if (total_size > 0) {
    DistributionInfo info = CalculateDistribution(total_size, world_size);
    info.local_count = info.sendcounts[world_rank];

    std::vector<int> local_data = ScatterVectorData(input_data_ptr, info, world_rank);

    local_min = ComputeLocalMinimum(local_data);
  }

  int total_min = PerformGlobalReduction(local_min);

  GetOutput() = total_min;
  return true;
}

bool ShkrylevaSVecMinValMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkryleva_s_vec_min_val
