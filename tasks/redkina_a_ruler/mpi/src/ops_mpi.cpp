#include "redkina_a_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"

namespace redkina_a_ruler {

namespace {

bool IsInStartEndChain(const int rank, const int start, const int end, const bool go_right) {
  if (go_right) {
    return (rank > start && rank <= end);
  }
  return (rank < start && rank >= end);
}

void SendSizeAndData(const int end_rank, const uint64_t data_size, const std::vector<int> &data, const int size_tag = 0,
                     const int data_tag = 1) {
  if (end_rank == MPI_PROC_NULL) {
    return;
  }

  MPI_Send(&data_size, 1, MPI_UINT64_T, end_rank, size_tag, MPI_COMM_WORLD);
  if (data_size > 0U) {
    MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, end_rank, data_tag, MPI_COMM_WORLD);
  }
}

void ReceiveSizeAndData(const int start_rank, uint64_t &data_size, std::vector<int> &data, const int size_tag = 0,
                        const int data_tag = 1) {
  if (start_rank == MPI_PROC_NULL) {
    return;
  }

  MPI_Recv(&data_size, 1, MPI_UINT64_T, start_rank, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (data_size > 0U) {
    data.resize(static_cast<std::size_t>(data_size));
    MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, start_rank, data_tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void BroadcastEndResult(const int rank, const int root, std::vector<int> &output) {
  uint64_t data_size = 0;

  if (rank == root) {
    data_size = static_cast<uint64_t>(output.size());
  }

  MPI_Bcast(&data_size, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  if (rank != root) {
    output.resize(static_cast<std::size_t>(data_size));
  }

  if (data_size > 0U) {
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, root, MPI_COMM_WORLD);
  }
}

void HandleSameStartEnd(const int rank, const int start, const std::vector<int> &input_data, std::vector<int> &output) {
  if (rank == start) {
    output = input_data;
  }

  auto data_size = static_cast<uint64_t>(input_data.size());
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, start, MPI_COMM_WORLD);

  if (rank != start) {
    output.resize(static_cast<std::size_t>(data_size));
  }

  if (data_size > 0U) {
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, start, MPI_COMM_WORLD);
  }
}

void RouteDataStartEnd(const int rank, const int start, const int end, const bool go_right, const int left_n,
                       const int right_n, const std::vector<int> &input_data, std::vector<int> &output) {
  std::vector<int> buffer;
  uint64_t data_size = 0;

  if (rank == start) {
    buffer = input_data;
    data_size = static_cast<uint64_t>(buffer.size());
    const int target = go_right ? right_n : left_n;
    SendSizeAndData(target, data_size, buffer);
  }

  if (!IsInStartEndChain(rank, start, end, go_right)) {
    return;
  }

  const int recv_from = go_right ? left_n : right_n;
  const int send_to = go_right ? right_n : left_n;

  ReceiveSizeAndData(recv_from, data_size, buffer);

  if (rank == end) {
    output = buffer;
  } else {
    SendSizeAndData(send_to, data_size, buffer);
  }
}

}  // namespace

RedkinaARulerMPI::RedkinaARulerMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaARulerMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.start >= 0 && input.end >= 0 && !input.data.empty();
}

bool RedkinaARulerMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RedkinaARulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const int start = input.start;
  const int end = input.end;

  if (start < 0 || start >= size || end < 0 || end >= size) {
    return false;
  }

  if (start == end) {
    HandleSameStartEnd(rank, start, input.data, GetOutput());
    return true;
  }

  const bool go_right = end > start;
  const int left_n = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right_n = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

  RouteDataStartEnd(rank, start, end, go_right, left_n, right_n, input.data, GetOutput());

  BroadcastEndResult(rank, end, GetOutput());
  return true;
}

bool RedkinaARulerMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_ruler
