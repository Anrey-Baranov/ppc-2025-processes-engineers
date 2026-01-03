#include "balchunayte_z_reduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace balchunayte_z_reduce {

namespace {

// Универсальное дерево-суммирование для типа T
template <typename T>
void TreeReduceSum(const T *send_buffer, T *recv_buffer, int count, int rank, int size, int root, MPI_Comm comm,
                   MPI_Datatype mpi_type) {
  const int message_tag = 0;
  MPI_Status status{};
  const int virtual_rank = (rank - root + size) % size;

  std::vector<T> accumulator(count);
  std::copy(send_buffer, send_buffer + count, accumulator.begin());
  std::vector<T> temp(count);

  for (int step_size = 1; step_size < size; step_size *= 2) {
    if (virtual_rank % (2 * step_size) == 0) {
      const int source_virtual_rank = virtual_rank + step_size;
      if (source_virtual_rank < size) {
        const int source_rank = (source_virtual_rank + root) % size;
        MPI_Recv(temp.data(), count, mpi_type, source_rank, message_tag, comm, &status);
        for (int index = 0; index < count; ++index) {
          accumulator[index] += temp[index];
        }
      }
    } else if (virtual_rank % step_size == 0) {
      const int target_virtual_rank = virtual_rank - step_size;
      const int target_rank = (target_virtual_rank + root) % size;
      MPI_Send(accumulator.data(), count, mpi_type, target_rank, message_tag, comm);
      return;
    }
  }

  if ((rank == root) && (recv_buffer != nullptr)) {
    std::ranges::copy(accumulator, recv_buffer);
  }
}

// Обёртка с нужной сигнатурой (как у MPI_Reduce, но только SUM + int/float/double)
int BalchunayteZReduce(const void *send_buffer, void *recv_buffer, int count, MPI_Datatype datatype, MPI_Op op,
                       int root, MPI_Comm comm) {
  if (count < 0) {
    return MPI_ERR_COUNT;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size <= 0) {
    return MPI_ERR_COMM;
  }

  if (op != MPI_SUM) {
    return MPI_ERR_OP;
  }

  if (count == 0) {
    return MPI_SUCCESS;
  }

  if (datatype == MPI_INT) {
    const auto *typed_send_buffer = static_cast<const int *>(send_buffer);
    auto *typed_recv_buffer = static_cast<int *>(recv_buffer);
    TreeReduceSum(typed_send_buffer, typed_recv_buffer, count, rank, size, root, comm, MPI_INT);
  } else if (datatype == MPI_FLOAT) {
    const auto *typed_send_buffer = static_cast<const float *>(send_buffer);
    auto *typed_recv_buffer = static_cast<float *>(recv_buffer);
    TreeReduceSum(typed_send_buffer, typed_recv_buffer, count, rank, size, root, comm, MPI_FLOAT);
  } else if (datatype == MPI_DOUBLE) {
    const auto *typed_send_buffer = static_cast<const double *>(send_buffer);
    auto *typed_recv_buffer = static_cast<double *>(recv_buffer);
    TreeReduceSum(typed_send_buffer, typed_recv_buffer, count, rank, size, root, comm, MPI_DOUBLE);
  } else {
    return MPI_ERR_TYPE;
  }

  return MPI_SUCCESS;
}

}  // namespace

bool BalchunayteZReduceMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int valid_flag = 1;

  if (world_rank_ == 0) {
    const auto &input = GetInput();

    if (input.data.empty()) {
      valid_flag = 0;
    }

    if ((input.root < 0) || (input.root >= world_size_)) {
      valid_flag = 0;
    }
  }

  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

bool BalchunayteZReduceMPI::PreProcessingImpl() {
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

  std::vector<int> counts(world_size_, 0);
  std::vector<int> displacements(world_size_, 0);

  const int base_size = global_size / world_size_;
  const int remainder = global_size % world_size_;
  for (int rank_index = 0; rank_index < world_size_; ++rank_index) {
    counts[rank_index] = base_size + ((rank_index < remainder) ? 1 : 0);
  }
  for (int rank_index = 1; rank_index < world_size_; ++rank_index) {
    displacements[rank_index] = displacements[rank_index - 1] + counts[rank_index - 1];
  }

  local_size_ = counts[world_rank_];
  local_data_.resize(local_size_);

  const double *send_data = nullptr;
  if (world_rank_ == 0) {
    send_data = GetInput().data.data();
  }

  MPI_Scatterv(send_data, counts.data(), displacements.data(), MPI_DOUBLE, local_data_.data(), local_size_, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  local_sum_ = 0.0;
  for (int index = 0; index < local_size_; ++index) {
    local_sum_ += local_data_[index];
  }

  GetOutput() = 0.0;
  return true;
}

bool BalchunayteZReduceMPI::RunImpl() {
  double global_sum = 0.0;

  const int reduce_status = BalchunayteZReduce(&local_sum_, (world_rank_ == root_) ? &global_sum : nullptr, 1,
                                               MPI_DOUBLE, MPI_SUM, root_, MPI_COMM_WORLD);
  if (reduce_status != MPI_SUCCESS) {
    return false;
  }

  const int message_tag = 1;

  if (world_rank_ == root_) {
    GetOutput() = global_sum;

    for (int rank_index = 0; rank_index < world_size_; ++rank_index) {
      if (rank_index == root_) {
        continue;
      }
      MPI_Send(&global_sum, 1, MPI_DOUBLE, rank_index, message_tag, MPI_COMM_WORLD);
    }
  } else {
    double received_value = 0.0;
    MPI_Status status{};
    MPI_Recv(&received_value, 1, MPI_DOUBLE, root_, message_tag, MPI_COMM_WORLD, &status);
    GetOutput() = received_value;
  }

  return true;
}

bool BalchunayteZReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace balchunayte_z_reduce
