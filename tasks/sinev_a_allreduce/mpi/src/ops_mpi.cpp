#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstring>
#include <variant>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_allreduce {

SinevAAllreduce::SinevAAllreduce(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool SinevAAllreduce::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized == 1;
}

bool SinevAAllreduce::PreProcessingImpl() {
  return true;
}

int SinevAAllreduce::GetTypeSize(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 1;
}

namespace {
template <typename T>
void PerformSumTemplate(T *out, const T *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}
}  // namespace

void SinevAAllreduce::PerformOperation(void *inout, const void *in, int count, MPI_Datatype datatype, MPI_Op op) {
  if (op != MPI_SUM) {
    return;
  }

  if (datatype == MPI_INT) {
    PerformSumTemplate(static_cast<int *>(inout), static_cast<const int *>(in), count);
  } else if (datatype == MPI_FLOAT) {
    PerformSumTemplate(static_cast<float *>(inout), static_cast<const float *>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    PerformSumTemplate(static_cast<double *>(inout), static_cast<const double *>(in), count);
  }
}

namespace {

void PerformReducePhase(int rank, int size, int total_bytes, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                        std::vector<char> &local_buffer) {
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;

    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> recv_buffer(total_bytes);
        MPI_Recv(recv_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
        SinevAAllreduce::PerformOperation(local_buffer.data(), recv_buffer.data(), count, datatype, op);
      } else {
        MPI_Send(local_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
}

void BroadcastViaBinaryTree(int rank, int size, int count, MPI_Datatype datatype, MPI_Comm comm, void *recvbuf) {
  int tree_size = 1;
  while (tree_size < size) {
    tree_size <<= 1;
  }

  for (int level = tree_size / 2; level > 0; level >>= 1) {
    if (rank < level) {
      int dest = rank + level;
      if (dest < size) {
        MPI_Send(recvbuf, count, datatype, dest, 1, comm);
      }
    } else if (rank < 2 * level && rank >= level) {
      int source = rank - level;
      if (source < size) {
        MPI_Recv(recvbuf, count, datatype, source, 1, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}

void BroadcastRemainingProcesses(int rank, int size, int count, MPI_Datatype datatype, MPI_Comm comm, void *recvbuf) {
  if (size <= 1) {
    return;
  }

  for (int step = 1; step < size; step *= 2) {
    if (rank < step) {
      int dest = rank + step;
      if (dest < size && dest >= step) {
        MPI_Send(recvbuf, count, datatype, dest, 2, comm);
      }
    } else if (rank < 2 * step && rank >= step) {
      int source = rank - step;
      if (source >= 0) {
        MPI_Recv(recvbuf, count, datatype, source, 2, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}

}  // namespace

int SinevAAllreduce::MpiAllreduceCustom(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                        MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size == 1) {
    int type_size = GetTypeSize(datatype);
    size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    std::memcpy(recvbuf, sendbuf, total_size);
    return 0;
  }

  int type_size = GetTypeSize(datatype);
  int total_bytes = count * type_size;

  std::vector<char> local_buffer(total_bytes);

  if (sendbuf == MPI_IN_PLACE) {
    std::memcpy(local_buffer.data(), recvbuf, total_bytes);
  } else {
    std::memcpy(local_buffer.data(), sendbuf, total_bytes);
  }

  PerformReducePhase(rank, size, total_bytes, count, datatype, op, comm, local_buffer);

  if (rank == 0) {
    std::memcpy(recvbuf, local_buffer.data(), total_bytes);
  }

  if (rank != 0 && sendbuf != MPI_IN_PLACE) {
    std::memcpy(recvbuf, sendbuf, total_bytes);
  }

  BroadcastViaBinaryTree(rank, size, count, datatype, comm, recvbuf);

  BroadcastRemainingProcesses(rank, size, count, datatype, comm, recvbuf);

  return 0;
}

bool SinevAAllreduce::RunImpl() {
  auto &input_variant = GetInput();
  auto &output_variant = GetOutput();

  try {
    if (std::holds_alternative<std::vector<int>>(input_variant)) {
      auto &input = std::get<std::vector<int>>(input_variant);
      auto &output = std::get<std::vector<int>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
      auto &input = std::get<std::vector<float>>(input_variant);
      auto &output = std::get<std::vector<float>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_FLOAT, MPI_SUM,
                         MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
      auto &input = std::get<std::vector<double>>(input_variant);
      auto &output = std::get<std::vector<double>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD);
    }

    return true;
  } catch (...) {
    return false;
  }
}

bool SinevAAllreduce::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce
