#include "alekseev_a_custom_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace alekseev_a_custom_reduce {

namespace {

template <typename T>
void ReduceTree(const T *send_data, T *recv_data, int count, int rank, int size, int root, MPI_Comm comm,
                MPI_Datatype mpi_type) {
  constexpr int kMessageTag = 0;
  const int virtual_rank = (rank - root + size) % size;

  std::vector<T> accumulator(send_data, send_data + count);
  std::vector<T> buffer(count);
  MPI_Status status;

  for (int step = 1; step < size; step <<= 1) {
    if ((virtual_rank & step) == 0) {
      const int source_virtual = virtual_rank + step;
      if (source_virtual < size) {
        const int source_rank = (source_virtual + root) % size;
        MPI_Recv(buffer.data(), count, mpi_type, source_rank, kMessageTag, comm, &status);

        for (int i = 0; i < count; ++i) {
          accumulator[i] += buffer[i];
        }
      }
    } else {
      const int dest_virtual = virtual_rank - step;
      const int dest_rank = (dest_virtual + root) % size;
      MPI_Send(accumulator.data(), count, mpi_type, dest_rank, kMessageTag, comm);
      return;
    }
  }

  if (rank == root && recv_data != nullptr) {
    std::copy(accumulator.begin(), accumulator.end(), recv_data);
  }
}

int CustomReduceImpl(const void *send_data, void *recv_data, int count, MPI_Datatype datatype, MPI_Op op, int root,
                     MPI_Comm comm) {
  if (count < 0) {
    return MPI_ERR_COUNT;
  }
  if (op != MPI_SUM) {
    return MPI_ERR_OP;
  }
  if (count == 0) {
    return MPI_SUCCESS;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size <= 0) {
    return MPI_ERR_COMM;
  }
  if (root < 0 || root >= size) {
    return MPI_ERR_ROOT;
  }

  if (datatype == MPI_INT) {
    ReduceTree(static_cast<const int *>(send_data), static_cast<int *>(recv_data), count, rank, size, root, comm,
               MPI_INT);
  } else if (datatype == MPI_FLOAT) {
    ReduceTree(static_cast<const float *>(send_data), static_cast<float *>(recv_data), count, rank, size, root, comm,
               MPI_FLOAT);
  } else if (datatype == MPI_DOUBLE) {
    ReduceTree(static_cast<const double *>(send_data), static_cast<double *>(recv_data), count, rank, size, root, comm,
               MPI_DOUBLE);
  } else {
    return MPI_ERR_TYPE;
  }

  return MPI_SUCCESS;
}

}  // namespace

bool AlekseevACustomReduceMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int validation_flag = 1;

  if (world_rank_ == 0) {
    const auto &input = GetInput();

    if (input.data.empty()) {
      validation_flag = 0;
    }

    if (input.root < 0 || input.root >= world_size_) {
      validation_flag = 0;
    }
  }

  MPI_Bcast(&validation_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return validation_flag == 1;
}

bool AlekseevACustomReduceMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  if (world_rank_ == 0) {
    root_ = GetInput().root;
  }
  MPI_Bcast(&root_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int global_size = 0;
  if (world_rank_ == 0) {
    global_size = static_cast<int>(GetInput().data.size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(world_size_);
  std::vector<int> displacements(world_size_, 0);

  const int base_chunk = global_size / world_size_;
  const int remainder = global_size % world_size_;

  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base_chunk + (i < remainder ? 1 : 0);
    if (i > 0) {
      displacements[i] = displacements[i - 1] + counts[i - 1];
    }
  }

  local_size_ = counts[world_rank_];
  local_data_.resize(local_size_);

  const double *global_data = world_rank_ == 0 ? GetInput().data.data() : nullptr;

  MPI_Scatterv(global_data, counts.data(), displacements.data(), MPI_DOUBLE, local_data_.data(), local_size_,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  local_sum_ = 0.0;
  for (double value : local_data_) {
    local_sum_ += value;
  }

  GetOutput() = 0.0;

  return true;
}

bool AlekseevACustomReduceMPI::RunImpl() {
  double global_sum = 0.0;
  double *recv_buffer = world_rank_ == root_ ? &global_sum : nullptr;

  const int reduce_result = CustomReduceImpl(&local_sum_, recv_buffer, 1, MPI_DOUBLE, MPI_SUM, root_, MPI_COMM_WORLD);

  if (reduce_result != MPI_SUCCESS) {
    return false;
  }

  if (world_rank_ == root_) {
    GetOutput() = global_sum;
  }

  return true;
}

bool AlekseevACustomReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace alekseev_a_custom_reduce
