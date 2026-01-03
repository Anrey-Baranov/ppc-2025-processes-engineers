#include "bortsova_a_transmission_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "bortsova_a_transmission_gather/common/include/common.hpp"

namespace bortsova_a_transmission_gather {

namespace {

void CopyReceivedData(std::vector<double> &gather_buffer, const std::vector<double> &recv_buffer,
                      const std::vector<int> &flags_int, std::vector<bool> &received, int world_size, int local_count) {
  for (int rank = 0; rank < world_size; ++rank) {
    if (flags_int[static_cast<std::size_t>(rank)] != 0) {
      std::size_t start_idx = static_cast<std::size_t>(rank) * static_cast<std::size_t>(local_count);
      for (int jj = 0; jj < local_count; ++jj) {
        gather_buffer[start_idx + static_cast<std::size_t>(jj)] = recv_buffer[start_idx + static_cast<std::size_t>(jj)];
      }
      received[static_cast<std::size_t>(rank)] = true;
    }
  }
}

void PrepareFlags(std::vector<int> &flags_int, const std::vector<bool> &received, int world_size) {
  for (int rank = 0; rank < world_size; ++rank) {
    flags_int[static_cast<std::size_t>(rank)] = static_cast<int>(received[static_cast<std::size_t>(rank)]);
  }
}

}  // namespace

BortsovaATransmissionGatherMPI::BortsovaATransmissionGatherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool BortsovaATransmissionGatherMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int root = GetInput().root;
  return root >= 0 && root < world_size_;
}

bool BortsovaATransmissionGatherMPI::PreProcessingImpl() {
  send_count_ = static_cast<int>(GetInput().send_data.size());
  return true;
}

bool BortsovaATransmissionGatherMPI::RunImpl() {
  int root = GetInput().root;
  const std::vector<double> &send_data = GetInput().send_data;

  int local_count = send_count_;
  MPI_Bcast(&local_count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (local_count == 0) {
    GetOutput().recv_data.clear();
    return true;
  }

  std::size_t total_size = static_cast<std::size_t>(world_size_) * static_cast<std::size_t>(local_count);
  std::vector<double> gather_buffer(total_size, 0.0);
  std::vector<bool> received(static_cast<std::size_t>(world_size_), false);

  std::size_t offset = static_cast<std::size_t>(world_rank_) * static_cast<std::size_t>(local_count);
  for (std::size_t idx = 0; idx < send_data.size(); ++idx) {
    gather_buffer[offset + idx] = send_data[idx];
  }
  received[static_cast<std::size_t>(world_rank_)] = true;

  TreeGather(gather_buffer, received, local_count, static_cast<int>(total_size));

  TransferToRoot(gather_buffer, root, static_cast<int>(total_size));

  MPI_Bcast(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, root, MPI_COMM_WORLD);

  GetOutput().recv_data = std::move(gather_buffer);

  return true;
}

void BortsovaATransmissionGatherMPI::TreeGather(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                int local_count, int total_size) {
  int step = 1;
  while (step < world_size_) {
    if ((world_rank_ % (2 * step)) == 0) {
      int source = world_rank_ + step;
      if (source < world_size_) {
        ReceiveFromChild(gather_buffer, received, source, local_count, total_size);
      }
    } else if ((world_rank_ % step) == 0) {
      SendToParent(gather_buffer, received, step, total_size);
      break;
    }
    step *= 2;
  }
}

void BortsovaATransmissionGatherMPI::ReceiveFromChild(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                      int source, int local_count, int total_size) const {
  std::vector<double> recv_buffer(static_cast<std::size_t>(total_size), 0.0);
  std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);

  MPI_Status status;
  MPI_Recv(recv_buffer.data(), total_size, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(flags_int.data(), world_size_, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

  CopyReceivedData(gather_buffer, recv_buffer, flags_int, received, world_size_, local_count);
}

void BortsovaATransmissionGatherMPI::SendToParent(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                  int step, int total_size) const {
  int dest = world_rank_ - step;
  MPI_Send(gather_buffer.data(), total_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

  std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);
  PrepareFlags(flags_int, received, world_size_);
  MPI_Send(flags_int.data(), world_size_, MPI_INT, dest, 1, MPI_COMM_WORLD);
}

void BortsovaATransmissionGatherMPI::TransferToRoot(std::vector<double> &gather_buffer, int root,
                                                    int total_size) const {
  if (world_rank_ == 0 && root != 0) {
    MPI_Send(gather_buffer.data(), total_size, MPI_DOUBLE, root, 2, MPI_COMM_WORLD);
  } else if (world_rank_ == root && root != 0) {
    MPI_Status status;
    MPI_Recv(gather_buffer.data(), total_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
  }
}

bool BortsovaATransmissionGatherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace bortsova_a_transmission_gather
