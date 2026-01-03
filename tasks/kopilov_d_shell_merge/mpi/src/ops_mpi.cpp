#include "kopilov_d_shell_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace kopilov_d_shell_merge {

namespace {

void ShellSort(std::vector<int> &vec) {
  const std::size_t n = vec.size();
  if (n < 2) {
    return;
  }
  for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = vec[i];
      std::size_t j = i;
      while (j >= gap && vec[j - gap] > tmp) {
        vec[j] = vec[j - gap];
        j -= gap;
      }
      vec[j] = tmp;
    }
  }
}

std::vector<int> SimpleMerge(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::ranges::merge(a, b, std::back_inserter(result));
  return result;
}

void SendVector(int dest, int tag, const std::vector<int> &vec, MPI_Comm comm) {
  auto size = static_cast<int>(vec.size());
  MPI_Send(&size, 1, MPI_INT, dest, tag, comm);
  MPI_Send(vec.data(), size, MPI_INT, dest, tag, comm);
}

std::vector<int> RecvVector(int source, int tag, MPI_Comm comm) {
  int size = 0;
  MPI_Status status;
  MPI_Recv(&size, 1, MPI_INT, source, tag, comm, &status);
  std::vector<int> vec(size);
  MPI_Recv(vec.data(), size, MPI_INT, source, tag, comm, &status);
  return vec;
}

}  // namespace

bool KopilovDShellMergeMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  return true;
}

bool KopilovDShellMergeMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int global_size = 0;
  if (world_rank_ == 0) {
    global_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  counts_.assign(world_size_, 0);
  displs_.assign(world_size_, 0);

  const int base = (global_size > 0) ? (global_size / world_size_) : 0;
  const int rem = (global_size > 0) ? (global_size % world_size_) : 0;

  for (int i = 0; i < world_size_; ++i) {
    counts_[i] = base + (i < rem ? 1 : 0);
  }
  for (int i = 1; i < world_size_; ++i) {
    displs_[i] = displs_[i - 1] + counts_[i - 1];
  }

  local_.resize(counts_[world_rank_]);

  MPI_Scatterv(GetInput().data(), counts_.data(), displs_.data(), MPI_INT, local_.data(),
               static_cast<int>(local_.size()), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().clear();
  return true;
}

bool KopilovDShellMergeMPI::RunImpl() {
  ShellSort(local_);
  return true;
}

bool KopilovDShellMergeMPI::PostProcessingImpl() {
  if (world_rank_ == 0) {
    for (int i = 1; i < world_size_; ++i) {
      std::vector<int> other = RecvVector(i, 0, MPI_COMM_WORLD);
      local_ = SimpleMerge(local_, other);
    }
    GetOutput() = std::move(local_);
    auto size = static_cast<int>(GetOutput().size());
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(GetOutput().data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    SendVector(0, 0, local_, MPI_COMM_WORLD);
    int size = 0;
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput().resize(size);
    MPI_Bcast(GetOutput().data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return true;
}

}  // namespace kopilov_d_shell_merge
