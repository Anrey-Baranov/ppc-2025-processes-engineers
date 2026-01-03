#include "zhurin_i_ring_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

namespace {

void SendData(int rank, int sender, int receiver, uint64_t data_size, const std::vector<int> &data) {
  if (rank != sender) {
    return;
  }

  MPI_Send(&data_size, 1, MPI_UINT64_T, receiver, 0, MPI_COMM_WORLD);
  if (data_size > 0) {
    MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, receiver, 1, MPI_COMM_WORLD);
  }
}

void ReceiveData(int rank, int sender, int receiver, uint64_t &data_size, std::vector<int> &buffer) {
  if (rank != receiver) {
    return;
  }

  MPI_Recv(&data_size, 1, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  buffer.resize(static_cast<size_t>(data_size));
  if (data_size > 0) {
    MPI_Recv(buffer.data(), static_cast<int>(data_size), MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void BroadcastToAll(int root, std::vector<int> &output) {
  auto data_size = static_cast<uint64_t>(output.size());
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  if (data_size > 0) {
    output.resize(static_cast<size_t>(data_size));
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, root, MPI_COMM_WORLD);
  }
}

}  // namespace

ZhurinIRingTopologyMPI::ZhurinIRingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZhurinIRingTopologyMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool ZhurinIRingTopologyMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZhurinIRingTopologyMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();

  int source = input.source % world_size;
  int dest = input.dest % world_size;

  if (source == dest) {
    if (rank == source) {
      GetOutput() = input.data;
    }
    BroadcastToAll(source, GetOutput());
    return true;
  }

  int direction = input.go_clockwise ? 1 : -1;
  int steps = 0;

  if (direction == 1) {
    steps = (dest - source + world_size) % world_size;
  } else {
    steps = (source - dest + world_size) % world_size;
  }

  std::vector<int> buffer;
  uint64_t data_size = 0;

  for (int step = 0; step < steps; step++) {
    int sender = (source + step * direction + world_size) % world_size;
    int receiver = (sender + direction + world_size) % world_size;

    if (step == 0) {
      SendData(rank, sender, receiver, static_cast<uint64_t>(input.data.size()), input.data);
    } else {
      SendData(rank, sender, receiver, data_size, buffer);
    }

    ReceiveData(rank, sender, receiver, data_size, buffer);

    if (receiver == dest && rank == dest) {
      GetOutput() = buffer;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  BroadcastToAll(dest, GetOutput());

  return true;
}

bool ZhurinIRingTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zhurin_i_ring_topology
