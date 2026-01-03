#include "tsyplakov_k_from_all_to_one/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tsyplakov_k_from_all_to_one {

template <typename T>
TsyplakovKFromAllToOneMPI<T>::TsyplakovKFromAllToOneMPI(const InTypeT<T> &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::ValidationImpl() {
  const auto &[data, root] = this->GetInput();
  return !data.empty() && root >= 0;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PreProcessingImpl() {
  gathered_.clear();
  return true;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::RunImpl() {
#ifdef USE_MPI
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[local_vec, root] = this->GetInput();
  const int sendcount = static_cast<int>(local_vec.size());

  MPI_Datatype mpi_type{};
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  } else {
    static_assert(!sizeof(T *), "Unsupported MPI type");
  }

  std::vector<T> recvbuf;
  if (rank == root) {
    recvbuf.resize(static_cast<std::size_t>(sendcount * size));
  }

  MyMpiGather(local_vec.data(), sendcount, mpi_type, rank == root ? recvbuf.data() : nullptr, sendcount, mpi_type, root,
              MPI_COMM_WORLD);

  if (rank == root) {
    gathered_ = std::move(recvbuf);
    this->GetOutput() = gathered_;
  }
#else
  this->GetOutput() = std::get<0>(this->GetInput());
#endif
  return true;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PostProcessingImpl() {
  return true;
}

namespace {

std::ptrdiff_t Offset(int index, int block_bytes) {
  return static_cast<std::ptrdiff_t>(index) * static_cast<std::ptrdiff_t>(block_bytes);
}

int CheckArgs(int sendcount, int recvcount, MPI_Datatype sendtype, MPI_Datatype recvtype) {
  if (sendcount != recvcount) {
    return MPI_ERR_COUNT;
  }
  if (sendtype != recvtype) {
    return MPI_ERR_TYPE;
  }
  return MPI_SUCCESS;
}

void ReceiveBlocks(int real_src, int block_bytes, MPI_Comm comm, std::vector<int> &ranks,
                   std::vector<std::byte> &data) {
  int recv_blocks = 0;
  MPI_Recv(&recv_blocks, 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);

  const std::size_t old_blocks = ranks.size();
  const std::size_t new_blocks = old_blocks + static_cast<std::size_t>(recv_blocks);

  ranks.resize(new_blocks);
  data.resize(new_blocks * static_cast<std::size_t>(block_bytes));

  for (int i = 0; i < recv_blocks; ++i) {
    const std::size_t idx = old_blocks + static_cast<std::size_t>(i);

    MPI_Recv(&ranks[idx], 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);

    MPI_Recv(data.data() + Offset(static_cast<int>(idx), block_bytes), block_bytes, MPI_BYTE, real_src, 0, comm,
             MPI_STATUS_IGNORE);
  }
}

void SendBlocks(int real_dest, int block_bytes, MPI_Comm comm, const std::vector<int> &ranks,
                const std::vector<std::byte> &data) {
  const int blocks = static_cast<int>(ranks.size());

  MPI_Send(&blocks, 1, MPI_INT, real_dest, 0, comm);

  for (int i = 0; i < blocks; ++i) {
    MPI_Send(&ranks[static_cast<std::size_t>(i)], 1, MPI_INT, real_dest, 0, comm);

    MPI_Send(data.data() + Offset(i, block_bytes), block_bytes, MPI_BYTE, real_dest, 0, comm);
  }
}

void GatherStep(int step, int size, int rel_rank, int root, int block_bytes, MPI_Comm comm, std::vector<int> &ranks,
                std::vector<std::byte> &data) {
  if (rel_rank % (2 * step) == 0) {
    const int src = rel_rank + step;
    if (src < size) {
      const int real_src = (src + root) % size;
      ReceiveBlocks(real_src, block_bytes, comm, ranks, data);
    }
  } else {
    const int dest = rel_rank - step;
    const int real_dest = (dest + root) % size;
    SendBlocks(real_dest, block_bytes, comm, ranks, data);
  }
}

void AssembleRoot(int block_bytes, void *recvbuf, const std::vector<int> &ranks, const std::vector<std::byte> &data) {
  const int blocks = static_cast<int>(ranks.size());

  for (int i = 0; i < blocks; ++i) {
    std::memcpy(static_cast<std::byte *>(recvbuf) + Offset(ranks[static_cast<std::size_t>(i)], block_bytes),
                data.data() + Offset(i, block_bytes), static_cast<std::size_t>(block_bytes));
  }
}

}  // namespace

int MyMpiGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  const int check = CheckArgs(sendcount, recvcount, sendtype, recvtype);
  if (check != MPI_SUCCESS) {
    return check;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int type_size = 0;
  MPI_Type_size(sendtype, &type_size);

  const int block_bytes = sendcount * type_size;

  std::vector<int> ranks(1, rank);
  std::vector<std::byte> data(static_cast<std::size_t>(block_bytes));

  std::memcpy(data.data(), sendbuf, static_cast<std::size_t>(block_bytes));

  const int rel_rank = (rank - root + size) % size;

  for (int step = 1; step < size; step <<= 1) {
    GatherStep(step, size, rel_rank, root, block_bytes, comm, ranks, data);
  }

  if (rank == root) {
    AssembleRoot(block_bytes, recvbuf, ranks, data);
  }

  return MPI_SUCCESS;
}

template class TsyplakovKFromAllToOneMPI<int>;
template class TsyplakovKFromAllToOneMPI<float>;
template class TsyplakovKFromAllToOneMPI<double>;

}  // namespace tsyplakov_k_from_all_to_one
