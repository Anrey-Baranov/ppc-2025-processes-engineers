#include "buzuluksky_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // std::ranges::merge, std::swap, std::copy_n
#include <cstddef>    // std::size_t, std::ptrdiff_t
#include <vector>

#include "buzuluksky_d_bubble_sort/common/include/common.hpp"

namespace buzuluksky_d_bubble_sort {

namespace {

void LocalOddEvenSort(std::vector<int> &data) {
  bool sorted = false;
  const std::size_t n = data.size();

  while (!sorted) {
    sorted = true;

    for (std::size_t i = 0; i + 1 < n; i += 2) {
      if (data[i] > data[i + 1]) {
        std::swap(data[i], data[i + 1]);
        sorted = false;
      }
    }

    for (std::size_t i = 1; i + 1 < n; i += 2) {
      if (data[i] > data[i + 1]) {
        std::swap(data[i], data[i + 1]);
        sorted = false;
      }
    }
  }
}

int PartnerRank(int rank, int phase) {
  const bool even_phase = (phase % 2) == 0;
  const bool even_rank = (rank % 2) == 0;

  if (even_phase) {
    return even_rank ? rank + 1 : rank - 1;
  }

  return even_rank ? rank - 1 : rank + 1;
}

void ExchangeWithNeighbor(std::vector<int> &local, int rank, int partner, const std::vector<int> &counts) {
  const int proc_count = static_cast<int>(counts.size());
  if (partner < 0 || partner >= proc_count || counts[rank] == 0 || counts[partner] == 0) {
    return;
  }

  std::vector<int> remote(static_cast<std::size_t>(counts[partner]));

  MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_INT, partner, 0, remote.data(),
               static_cast<int>(remote.size()), MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(local.size() + remote.size());
  std::ranges::merge(local, remote, merged.begin());

  const std::size_t local_size = local.size();
  if (rank < partner) {
    std::copy_n(merged.begin(), local_size, local.begin());
  } else {
    std::copy_n(merged.end() - static_cast<std::ptrdiff_t>(local_size), local_size, local.begin());
  }
}

}  // namespace

BuzulukskyDBubbleSortMPI::BuzulukskyDBubbleSortMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  static_cast<void>(GetOutput());
}

bool BuzulukskyDBubbleSortMPI::ValidationImpl() {
  return true;
}

bool BuzulukskyDBubbleSortMPI::PreProcessingImpl() {
  return true;
}

bool BuzulukskyDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int proc_count = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  const auto &input = GetInput();
  const int n = static_cast<int>(input.size());

  std::vector<int> counts(proc_count, 0);
  std::vector<int> displs(proc_count, 0);

  if (rank == 0) {
    int offset = 0;
    const int base = n / proc_count;
    const int remainder = n % proc_count;

    for (int i = 0; i < proc_count; ++i) {
      counts[i] = base + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += counts[i];
    }
  }

  MPI_Bcast(counts.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local(static_cast<std::size_t>(counts[rank]));

  MPI_Scatterv(rank == 0 ? input.data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(), counts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  LocalOddEvenSort(local);

  for (int phase = 0; phase < proc_count + 1; ++phase) {
    const int partner = PartnerRank(rank, phase);
    ExchangeWithNeighbor(local, rank, partner, counts);
  }

  std::vector<int> result(static_cast<std::size_t>(n));

  MPI_Gatherv(local.data(), counts[rank], MPI_INT, rank == 0 ? result.data() : nullptr, counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().resize(n);
  if (rank == 0) {
    GetOutput() = result;
  }

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool BuzulukskyDBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace buzuluksky_d_bubble_sort
