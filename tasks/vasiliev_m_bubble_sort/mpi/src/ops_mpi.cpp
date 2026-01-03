#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

VasilievMBubbleSortMPI::VasilievMBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBubbleSortMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMBubbleSortMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &vec = GetInput();
  int n = static_cast<int>(vec.size());

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalcCountsAndDispls(n, size, counts, displs);
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  BubbleOESort(local_data);

  for (int phase = 0; phase < size + 1; phase++) {
    int partner = FindPartner(rank, phase);

    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(rank, partner, local_data);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().resize(n);
  }

  MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(n);
  }

  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void VasilievMBubbleSortMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;

  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }

  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

void VasilievMBubbleSortMPI::BubbleOESort(std::vector<int> &vec) {
  const size_t n = vec.size();
  bool sorted = false;

  if (vec.empty()) {
    return;
  }

  while (!sorted) {
    sorted = true;

    for (size_t i = 0; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }
  }
}

int VasilievMBubbleSortMPI::FindPartner(int rank, int phase) {
  if (phase % 2 == 0) {
    return (rank % 2 == 0) ? rank + 1 : rank - 1;
  }
  return (rank % 2 == 0) ? rank - 1 : rank + 1;
}

void VasilievMBubbleSortMPI::ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data) {
  int local_size = static_cast<int>(local_data.size());
  int partner_size = 0;

  MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> partner_data(partner_size);

  MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 1, partner_data.data(), partner_size, MPI_INT, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(local_size + partner_size);
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < local_data.size() && j < partner_data.size()) {
    merged[k++] = (local_data[i] <= partner_data[j]) ? local_data[i++] : partner_data[j++];
  }
  while (i < local_data.size()) {
    merged[k++] = local_data[i++];
  }
  while (j < partner_data.size()) {
    merged[k++] = partner_data[j++];
  }

  if (rank < partner) {
    std::copy(merged.begin(), merged.begin() + local_size, local_data.begin());
  } else {
    std::copy(merged.end() - local_size, merged.end(), local_data.begin());
  }
}

bool VasilievMBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bubble_sort
