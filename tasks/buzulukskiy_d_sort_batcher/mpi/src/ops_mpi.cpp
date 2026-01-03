#include "buzulukskiy_d_sort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

#include "buzulukskiy_d_sort_batcher/common/include/common.hpp"

namespace buzulukskiy_d_sort_batcher {
namespace {
constexpr int kRadixBase = 10;
constexpr int kMaxIterations = 100;

void RadixSortUnsigned(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }
  const int max_val = *std::ranges::max_element(arr);
  std::vector<int> output(arr.size());
  for (int exp = 1; max_val / exp > 0; exp *= kRadixBase) {
    std::vector<int> count(static_cast<std::size_t>(kRadixBase), 0);
    for (const int val : arr) {
      count[static_cast<std::size_t>((val / exp) % kRadixBase)]++;
    }
    for (std::size_t i = 1; i < static_cast<std::size_t>(kRadixBase); ++i) {
      count[i] += count[i - 1];
    }
    for (std::size_t i = arr.size(); i-- > 0;) {
      const int digit = (arr[i] / exp) % kRadixBase;
      output[--count[static_cast<std::size_t>(digit)]] = arr[i];
    }
    arr.swap(output);
  }
}

void RadixSortLSD(std::vector<int> &data) {
  if (data.empty()) {
    return;
  }
  std::vector<int> positives;
  std::vector<int> negatives;
  for (const int val : data) {
    if (val < 0) {
      negatives.push_back(val == INT_MIN ? INT_MAX : -val);
    } else {
      positives.push_back(val);
    }
  }
  if (!positives.empty()) {
    RadixSortUnsigned(positives);
  }
  if (!negatives.empty()) {
    RadixSortUnsigned(negatives);
    std::ranges::reverse(negatives);
    for (int &v_ref : negatives) {
      v_ref = (v_ref == INT_MAX ? INT_MIN : -v_ref);
    }
  }
  data.clear();
  data.insert(data.end(), negatives.begin(), negatives.end());
  data.insert(data.end(), positives.begin(), positives.end());
}

void ExchangeAndMerge(std::vector<int> &local, int partner, const std::vector<int> &counts, int rank) {
  if (partner < 0 || std::cmp_greater_equal(partner, counts.size())) {
    return;
  }
  std::vector<int> remote(static_cast<std::size_t>(counts[static_cast<std::size_t>(partner)]));
  MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_INT, partner, 0, remote.data(),
               static_cast<int>(remote.size()), MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> combined;
  combined.reserve(local.size() + remote.size());
  std::ranges::merge(local, remote, std::back_inserter(combined));
  const auto my_size = static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)]);
  if (rank < partner) {
    local.assign(combined.begin(), combined.begin() + static_cast<std::ptrdiff_t>(my_size));
  } else {
    local.assign(combined.end() - static_cast<std::ptrdiff_t>(my_size), combined.end());
  }
}

void BatcherStep(int i, int j, int k, int phase_step, int rank, int size, std::vector<int> &local,
                 const std::vector<int> &counts) {
  const int r1 = i + j;
  const int r2 = i + j + k;
  if (r2 < size && (r1 / (phase_step * 2)) == (r2 / (phase_step * 2))) {
    if (rank == r1) {
      ExchangeAndMerge(local, r2, counts, rank);
    } else if (rank == r2) {
      ExchangeAndMerge(local, r1, counts, rank);
    }
  }
}

void BatcherInner(int k, int phase_step, int rank, int size, std::vector<int> &local, const std::vector<int> &counts) {
  for (int j = k % phase_step; j + k < size; j += 2 * k) {
    for (int i = 0; i < k; ++i) {
      BatcherStep(i, j, k, phase_step, rank, size, local, counts);
    }
  }
}

void BatcherNetworkPhase(std::vector<int> &local, int rank, int size, const std::vector<int> &counts) {
  // Исправлено: p -> phase_step (длина имени)
  for (int phase_step = 1; phase_step < size; phase_step <<= 1) {
    for (int k = phase_step; k > 0; k >>= 1) {
      BatcherInner(k, phase_step, rank, size, local, counts);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void BatcherStabilizationPhase(std::vector<int> &local, int rank, int size, const std::vector<int> &counts) {
  const int step_limit = std::min(size, kMaxIterations);
  for (int step_idx = 0; step_idx < step_limit; ++step_idx) {
    bool even_step = (step_idx % 2 == 0);
    bool even_rank = (rank % 2 == 0);
    int partner = (even_step == even_rank) ? rank + 1 : rank - 1;
    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(local, partner, counts, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::size_t> CalculateDistribution(int n, int size, int rank) {
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int base = n / size;
  int rem = n % size;
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[static_cast<std::size_t>(i)] = base + (i < rem ? 1 : 0);
    displs[static_cast<std::size_t>(i)] = offset;
    offset += counts[static_cast<std::size_t>(i)];
  }
  return {counts, displs, static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)])};
}
}  // namespace

BuzulukskiyDSortBatcherMPI::BuzulukskiyDSortBatcherMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BuzulukskiyDSortBatcherMPI::ValidationImpl() {
  return true;
}
bool BuzulukskiyDSortBatcherMPI::PreProcessingImpl() {
  return true;
}
bool BuzulukskiyDSortBatcherMPI::PostProcessingImpl() {
  return true;
}

bool BuzulukskiyDSortBatcherMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int n = (rank == 0) ? static_cast<int>(GetInput().size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n == 0) {
    if (rank == 0) {
      GetOutput() = InType();
    }
    return true;
  }
  auto [counts, displs, local_size] = CalculateDistribution(n, size, rank);
  std::vector<int> local(local_size);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(),
               static_cast<int>(local_size), MPI_INT, 0, MPI_COMM_WORLD);
  if (!local.empty()) {
    RadixSortLSD(local);
  }
  BatcherNetworkPhase(local, rank, size, counts);
  BatcherStabilizationPhase(local, rank, size, counts);
  std::vector<int> res(static_cast<std::size_t>(n));
  MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_INT, res.data(), counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);
  GetOutput() = std::move(res);
  return true;
}
}  // namespace buzulukskiy_d_sort_batcher
