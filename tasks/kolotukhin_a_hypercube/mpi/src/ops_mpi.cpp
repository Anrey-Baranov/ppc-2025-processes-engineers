#include "kolotukhin_a_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "kolotukhin_a_hypercube/common/include/common.hpp"

namespace kolotukhin_a_hypercube {

KolotukhinAHypercubeMPI::KolotukhinAHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

int KolotukhinAHypercubeMPI::GetNeighbor(int rank, int dim) {
  int neighbor = rank ^ (1 << dim);
  return neighbor;
}

int KolotukhinAHypercubeMPI::CalculateHypercubeDimension(int num_processes) {
  if (num_processes <= 1) {
    return 0;
  }
  int dimension = 0;
  int capacity = 1;
  while (capacity < num_processes) {
    dimension++;
    capacity *= 2;
  }
  return dimension;
}

void KolotukhinAHypercubeMPI::PerformComputeLoad(int iterations) {
  volatile int dummy = 0;
  for (int i = 0; i < iterations; i++) {
    dummy += (i * 3) / 7;
    dummy ^= (i << 3);
    dummy = dummy % 10007;
  }
  [[maybe_unused]] int final_value = dummy;
}

void KolotukhinAHypercubeMPI::SendData(std::vector<int> &data, int next_neighbor) {
  int data_size = static_cast<int>(data.size());
  MPI_Send(&data_size, 1, MPI_INT, next_neighbor, 0, MPI_COMM_WORLD);
  if (data_size > 0) {
    MPI_Send(data.data(), data_size, MPI_INT, next_neighbor, 1, MPI_COMM_WORLD);
  }
}

void KolotukhinAHypercubeMPI::RecvData(std::vector<int> &data, int prev_neighbor) {
  int data_size = 0;
  MPI_Recv(&data_size, 1, MPI_INT, prev_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  data.resize(static_cast<size_t>(data_size));
  if (data_size > 0) {
    MPI_Recv(data.data(), data_size, MPI_INT, prev_neighbor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void KolotukhinAHypercubeMPI::CalcPositions(int my_rank, std::vector<int> &path, int &my_pos, int &next, int &prev) {
  for (int i = 0; std::cmp_less(i, static_cast<int>(path.size())); i++) {
    if (my_rank == path[i]) {
      my_pos = i;
      if (i > 0) {
        prev = path[i - 1];
      }
      if (std::cmp_less(i, static_cast<int>(path.size() - 1))) {
        next = path[i + 1];
      }
      break;
    }
  }
}

std::vector<int> KolotukhinAHypercubeMPI::CalcPathLowToHigh(int source, int dest, int dimensions, int xor_val) {
  std::vector<int> path;
  int current = source;
  path.push_back(current);
  for (int dim = 0; dim < dimensions; dim++) {
    int mask = 1 << dim;
    if ((xor_val & mask) != 0) {
      current = current ^ mask;
      path.push_back(current);
      if (current == dest) {
        break;
      }
    }
  }
  return path;
}

std::vector<int> KolotukhinAHypercubeMPI::CalcPathHighToLow(int source, int dest, int dimensions, int xor_val) {
  std::vector<int> path;
  int current = source;
  path.push_back(current);
  for (int dim = dimensions - 1; dim >= 0; dim--) {
    int mask = 1 << dim;
    if ((xor_val & mask) != 0) {
      current = current ^ mask;
      path.push_back(current);
      if (current == dest) {
        break;
      }
    }
  }
  return path;
}

std::vector<int> KolotukhinAHypercubeMPI::CalcPath(int source, int dest, int dimensions) {
  int xor_val = source ^ dest;
  if (source > dest) {
    return CalcPathHighToLow(source, dest, dimensions, xor_val);
  }
  return CalcPathLowToHigh(source, dest, dimensions, xor_val);
}

bool KolotukhinAHypercubeMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto &input = GetInput();
  if ((input[0] < 0) || (input[0] > world_size - 1) || ((input[1] < 0) && (input[1] != -2)) ||
      (input[1] > world_size - 1) || (world_size <= 0)) {
    input[0] = 0;
    input[1] = world_size - 1;
  }
  return true;
}

bool KolotukhinAHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KolotukhinAHypercubeMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input[0];
  int dest = input[1];

  std::vector<int> data{};
  int data_size = 0;

  int dimensions = 0;
  dimensions = CalculateHypercubeDimension(world_size);
  if (rank == source) {
    data_size = input[2];
    data.resize(static_cast<size_t>(data_size), 1);
  }

  if (source == dest) {
    MPI_Bcast(&data_size, 1, MPI_INT, dest, MPI_COMM_WORLD);
    if (rank != dest) {
      data.resize(static_cast<size_t>(data_size));
    }
    MPI_Bcast(data.data(), data_size, MPI_INT, dest, MPI_COMM_WORLD);
    GetOutput() = std::accumulate(data.begin(), data.end(), 0);
    return true;
  }

  std::vector<int> path = CalcPath(source, dest, dimensions);

  int my_position = -1;
  int prev_neighbor = -1;
  int next_neighbor = -1;
  CalcPositions(rank, path, my_position, next_neighbor, prev_neighbor);
  if (my_position != -1) {
    if (rank == source) {
      PerformComputeLoad(150000);
      SendData(data, next_neighbor);
    } else if (rank == dest) {
      RecvData(data, prev_neighbor);
      data_size = static_cast<int>(data.size());
      PerformComputeLoad(150000);
    } else {
      RecvData(data, prev_neighbor);
      data_size = static_cast<int>(data.size());
      PerformComputeLoad(150000);
      SendData(data, next_neighbor);
    }
  }

  MPI_Bcast(&data_size, 1, MPI_INT, dest, MPI_COMM_WORLD);

  if (my_position == -1) {
    data.resize(static_cast<size_t>(data_size));
  }

  MPI_Bcast(data.data(), data_size, MPI_INT, dest, MPI_COMM_WORLD);

  GetOutput() = std::accumulate(data.begin(), data.end(), 0);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KolotukhinAHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_hypercube
