#include "kondrashova_v_ring_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cassert>
#include <vector>

#include "kondrashova_v_ring_topology/common/include/common.hpp"

namespace kondrashova_v_ring_topology {

KondrashovaVRingTopologyMPI::KondrashovaVRingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KondrashovaVRingTopologyMPI::ValidationImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int is_valid = 1;

  if (rank == 0) {
    const auto &input = GetInput();

    if (input.source < 0 || input.source >= world_size) {
      is_valid = 0;
    }
    if (input.recipient < 0 || input.recipient >= world_size) {
      is_valid = 0;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return is_valid == 1;
}

bool KondrashovaVRingTopologyMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void KondrashovaVRingTopologyMPI::SendData(int rank, int sender, int receiver, int step, int data_size,
                                           const std::vector<int> &data, const std::vector<int> &buffer) {
  if (rank != sender) {
    return;
  }

  if (step == 0) {
    MPI_Send(&data_size, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    if (data_size > 0) {
      MPI_Send(data.data(), data_size, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    }
  } else {
    int buf_size = static_cast<int>(buffer.size());
    MPI_Send(&buf_size, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    if (buf_size > 0) {
      MPI_Send(buffer.data(), buf_size, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    }
  }
}

void KondrashovaVRingTopologyMPI::ReceiveData(int rank, int sender, int receiver, int recipient,
                                              std::vector<int> &buffer) {
  if (rank != receiver) {
    return;
  }

  int recv_size = 0;
  MPI_Recv(&recv_size, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  buffer.resize(recv_size);
  if (recv_size > 0) {
    MPI_Recv(buffer.data(), recv_size, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == recipient) {
    GetOutput() = buffer;
  }
}

bool KondrashovaVRingTopologyMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int source = 0;
  int recipient = 0;
  int data_size = 0;

  if (rank == 0) {
    source = GetInput().source;
    recipient = GetInput().recipient;
    data_size = static_cast<int>(GetInput().data.size());
  }

  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&recipient, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> data(data_size);
  if (rank == 0) {
    data = GetInput().data;
  }
  if (data_size > 0) {
    MPI_Bcast(data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (world_size == 1 || source == recipient) {
    if (rank == recipient) {
      GetOutput() = data;
    }
    return true;
  }

  int steps = (recipient - source + world_size) % world_size;

  std::vector<int> buffer;

  for (int step = 0; step < steps; step++) {
    int sender = (source + step) % world_size;
    int receiver = (sender + 1) % world_size;

    SendData(rank, sender, receiver, step, data_size, data, buffer);
    ReceiveData(rank, sender, receiver, recipient, buffer);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  return true;
}

bool KondrashovaVRingTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondrashova_v_ring_topology
