#include "tabalaev_a_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <climits>
#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"

namespace tabalaev_a_linear_topology {

TabalaevALinearTopologyMPI::TabalaevALinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevALinearTopologyMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int sender = std::get<0>(GetInput());

  int validation = 0;
  if (world_rank == sender) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    auto receiver = std::get<1>(GetInput());
    auto data = std::get<2>(GetInput());

    if (sender >= 0 && sender < world_size && receiver >= 0 && receiver < world_size && !data.empty()) {
      validation = 1;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, sender, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevALinearTopologyMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool TabalaevALinearTopologyMPI::RunImpl() {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto sender = std::get<0>(GetInput());
  auto receiver = std::get<1>(GetInput());

  if (sender == receiver) {
    GetOutput() = std::get<2>(GetInput());
    return true;
  }

  int left = GetLeft(world_rank);
  int right = GetRight(world_rank, world_size);
  int direction = GetDirection(sender, receiver);

  std::vector<int> local_buff;

  if (world_rank == sender) {
    ProcessSender(direction, left, right, local_buff);
  } else if (world_rank == receiver) {
    ProcessReceiver(direction, left, right, local_buff);
  } else if (IsOnPath(world_rank, sender, receiver, direction)) {
    ProcessIntermediate(direction, left, right, local_buff);
  }

  int data_size = (world_rank == receiver) ? static_cast<int>(local_buff.size()) : 0;
  MPI_Bcast(&data_size, 1, MPI_INT, receiver, MPI_COMM_WORLD);

  if (world_rank != receiver) {
    local_buff.resize(data_size);
  }
  MPI_Bcast(local_buff.data(), data_size, MPI_INT, receiver, MPI_COMM_WORLD);

  GetOutput() = local_buff;

  return true;
}

bool TabalaevALinearTopologyMPI::PostProcessingImpl() {
  return true;
}

int TabalaevALinearTopologyMPI::GetLeft(int rank) {
  return rank == 0 ? MPI_PROC_NULL : rank - 1;
}

int TabalaevALinearTopologyMPI::GetRight(int rank, int size) {
  return rank == (size - 1) ? MPI_PROC_NULL : rank + 1;
}

int TabalaevALinearTopologyMPI::GetDirection(int sender, int receiver) {
  return sender < receiver ? 1 : -1;
}

bool TabalaevALinearTopologyMPI::IsOnPath(int rank, int sender, int receiver, int direction) {
  if (direction == 1) {
    return rank > sender && rank < receiver;
  }
  return rank < sender && rank > receiver;
}

void TabalaevALinearTopologyMPI::ProcessSender(int direction, int left, int right, std::vector<int> &local_buff) {
  local_buff = std::get<2>(GetInput());

  int size = static_cast<int>(local_buff.size());
  int to = (direction == 1 ? right : left);

  MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
}

void TabalaevALinearTopologyMPI::ProcessIntermediate(int direction, int left, int right, std::vector<int> &local_buff) {
  int from = (direction == 1 ? left : right);
  int to = (direction == 1 ? right : left);

  int size = 0;
  MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  local_buff.resize(size);
  MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
}

void TabalaevALinearTopologyMPI::ProcessReceiver(int direction, int left, int right, std::vector<int> &local_buff) {
  int from = (direction == 1 ? left : right);

  int size = 0;
  MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  local_buff.resize(size);
  MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

}  // namespace tabalaev_a_linear_topology
