#include "yakimov_i_linear_virtual_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"

namespace yakimov_i_linear_virtual_topology {

namespace {
constexpr int kMaxProcesses = 20;

bool IsValidProcess(int process_id) {
  bool result = false;
  result = (process_id >= 0) && (process_id < kMaxProcesses);
  return result;
}

void ProcessTriplet(int sender, int receiver, int data_value, int rank, int &local_sum) {
  if (IsValidProcess(sender) && IsValidProcess(receiver)) {
    if (rank == receiver) {
      local_sum += data_value;
    }
  }
}
}  // namespace

YakimovILinearVirtualTopologyMPI::YakimovILinearVirtualTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  std::filesystem::path base_path = std::filesystem::current_path();
  while (base_path.filename() != "ppc-2025-processes-engineers") {
    base_path = base_path.parent_path();
  }
  data_filename_ =
      base_path.string() + "/tasks/yakimov_i_linear_virtual_topology/data/" + std::to_string(GetInput()) + ".txt";
}

bool YakimovILinearVirtualTopologyMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    bool result = false;
    result = (GetInput() > 0);
    return result;
  }
  return true;
}

bool YakimovILinearVirtualTopologyMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    ReadDataFromFile(data_filename_);
  }

  local_sum_ = 0;
  total_sum_ = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

void YakimovILinearVirtualTopologyMPI::ReadDataFromFile(const std::string &filename) {
  std::ifstream file(filename);

  int value = 0;
  data_.clear();

  while (file >> value) {
    data_.push_back(value);
  }

  file.close();
}

void YakimovILinearVirtualTopologyMPI::CreateLinearTopology(MPI_Comm &linear_comm) {
  MPI_Comm_dup(MPI_COMM_WORLD, &linear_comm);
}

void YakimovILinearVirtualTopologyMPI::ProcessDataInTopology(int rank, MPI_Comm &linear_comm) {
  int rank_in_linear = 0;
  int size_in_linear = 0;
  MPI_Comm_rank(linear_comm, &rank_in_linear);
  MPI_Comm_size(linear_comm, &size_in_linear);

  for (size_t i = 0; i + 2 < data_.size(); i += 3) {
    int sender = data_[i];
    int receiver = data_[i + 1];
    int data_value = data_[i + 2];

    ProcessTriplet(sender, receiver, data_value, rank, local_sum_);
  }

  MPI_Barrier(linear_comm);
}

void YakimovILinearVirtualTopologyMPI::ExchangeDataInTopology(MPI_Comm &linear_comm) {
  int all_sum = 0;
  MPI_Allreduce(&local_sum_, &all_sum, 1, MPI_INT, MPI_SUM, linear_comm);
  total_sum_ = all_sum;

  MPI_Comm_free(&linear_comm);
}

bool YakimovILinearVirtualTopologyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int data_size = 0;
  if (rank == 0) {
    data_size = static_cast<int>(data_.size());
  }
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    data_.resize(static_cast<size_t>(data_size));
  }

  MPI_Bcast(data_.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Comm linear_comm = MPI_COMM_NULL;
  CreateLinearTopology(linear_comm);
  ProcessDataInTopology(rank, linear_comm);
  ExchangeDataInTopology(linear_comm);

  return true;
}

bool YakimovILinearVirtualTopologyMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = total_sum_;
  }

  OutType final_result = GetOutput();
  MPI_Bcast(&final_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = final_result;

  return true;
}

}  // namespace yakimov_i_linear_virtual_topology
