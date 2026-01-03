#include "makoveeva_s_cannon_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "makoveeva_s_cannon_algorithm/common/include/common.hpp"

namespace makoveeva_s_cannon_algorithm {

namespace {

bool CheckInputLocal(const std::vector<double> &a, const std::vector<double> &b, int n) {
  if (n <= 0) {
    return false;
  }
  const auto n_sz = static_cast<std::size_t>(n);
  const auto expected = n_sz * n_sz;
  return (a.size() == expected) && (b.size() == expected);
}

int ChooseGridQ(int size, int n) {
  if (size <= 0 || n <= 0) {
    return 0;
  }
  int q = static_cast<int>(std::floor(std::sqrt(static_cast<double>(size))));
  while (q > 0) {
    if ((q * q) <= size && (n % q == 0)) {
      return q;
    }
    --q;
  }
  return 0;
}

int BroadcastN(const InType &input, int world_rank) {
  int n = 0;
  if (world_rank == 0) {
    n = std::get<2>(input);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return n;
}

MPI_Datatype MakeBlockType(int n, int bs) {
  MPI_Datatype block = MPI_DATATYPE_NULL;
  MPI_Type_vector(bs, bs, n, MPI_DOUBLE, &block);

  MPI_Datatype resized = MPI_DATATYPE_NULL;
  MPI_Type_create_resized(block, 0, static_cast<MPI_Aint>(sizeof(double)), &resized);

  MPI_Type_commit(&resized);
  MPI_Type_free(&block);
  return resized;
}

void LocalMatMulAcc(const std::vector<double> &a, const std::vector<double> &b, int bs, std::vector<double> *c) {
  const auto bs_sz = static_cast<std::size_t>(bs);
  for (int i = 0; i < bs; ++i) {
    const auto i_sz = static_cast<std::size_t>(i);
    for (int k = 0; k < bs; ++k) {
      const auto k_sz = static_cast<std::size_t>(k);
      const double a_ik = a[(i_sz * bs_sz) + k_sz];
      for (int j = 0; j < bs; ++j) {
        const auto j_sz = static_cast<std::size_t>(j);
        (*c)[(i_sz * bs_sz) + j_sz] += a_ik * b[(k_sz * bs_sz) + j_sz];
      }
    }
  }
}

MPI_Comm MakeActiveComm(int world_rank, int active_p, bool *is_active) {
  *is_active = (world_rank < active_p);
  MPI_Comm active_comm = MPI_COMM_NULL;
  const int color = (*is_active) ? 0 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active_comm);
  return active_comm;
}

MPI_Comm MakeCartComm(MPI_Comm active_comm, int q) {
  const std::array<int, 2> dims = {q, q};
  const std::array<int, 2> periods = {1, 1};

  MPI_Comm cart_comm = MPI_COMM_NULL;
  MPI_Cart_create(active_comm, 2, dims.data(), periods.data(), 0 /*reorder*/, &cart_comm);
  return cart_comm;
}

std::array<int, 2> GetCoords(MPI_Comm cart_comm) {
  int cart_rank = 0;
  MPI_Comm_rank(cart_comm, &cart_rank);

  std::array<int, 2> coords = {0, 0};
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords.data());
  return coords;
}

void BuildScatterMeta(int q, int bs, int n, int active_p, std::vector<int> *sendcounts, std::vector<int> *displs,
                      MPI_Comm cart_comm) {
  int cart_rank = 0;
  MPI_Comm_rank(cart_comm, &cart_rank);

  if (cart_rank != 0) {
    return;
  }

  sendcounts->assign(static_cast<std::size_t>(active_p), 1);
  displs->assign(static_cast<std::size_t>(active_p), 0);

  for (int row_idx = 0; row_idx < q; ++row_idx) {
    for (int col_idx = 0; col_idx < q; ++col_idx) {
      const int proc = (row_idx * q) + col_idx;
      (*displs)[static_cast<std::size_t>(proc)] = ((row_idx * bs) * n) + (col_idx * bs);
    }
  }
}

void ScatterBlocks(const double *a_full_ptr, const double *b_full_ptr, const int *sendcounts_ptr, const int *displs_ptr,
                   MPI_Datatype block_type, int bs, MPI_Comm cart_comm, std::vector<double> *a_block,
                   std::vector<double> *b_block) {
  MPI_Scatterv(a_full_ptr, sendcounts_ptr, displs_ptr, block_type, a_block->data(), bs * bs, MPI_DOUBLE, 0, cart_comm);
  MPI_Scatterv(b_full_ptr, sendcounts_ptr, displs_ptr, block_type, b_block->data(), bs * bs, MPI_DOUBLE, 0, cart_comm);
}

void InitialShift(MPI_Comm cart_comm, int row, int col, int bs, std::vector<double> *a_block,
                  std::vector<double> *b_block) {
  int src = 0;
  int dst = 0;

  MPI_Cart_shift(cart_comm, 1, -row, &src, &dst);
  MPI_Sendrecv_replace(a_block->data(), bs * bs, MPI_DOUBLE, dst, 0, src, 0, cart_comm, MPI_STATUS_IGNORE);

  MPI_Cart_shift(cart_comm, 0, -col, &src, &dst);
  MPI_Sendrecv_replace(b_block->data(), bs * bs, MPI_DOUBLE, dst, 1, src, 1, cart_comm, MPI_STATUS_IGNORE);
}

void CannonLoop(MPI_Comm cart_comm, int q, int bs, std::vector<double> *a_block, std::vector<double> *b_block,
                std::vector<double> *c_block) {
  int src = 0;
  int dst = 0;

  for (int step = 0; step < q; ++step) {
    LocalMatMulAcc(*a_block, *b_block, bs, c_block);

    if (step + 1 < q) {
      MPI_Cart_shift(cart_comm, 1, -1, &src, &dst);
      MPI_Sendrecv_replace(a_block->data(), bs * bs, MPI_DOUBLE, dst, 2, src, 2, cart_comm, MPI_STATUS_IGNORE);

      MPI_Cart_shift(cart_comm, 0, -1, &src, &dst);
      MPI_Sendrecv_replace(b_block->data(), bs * bs, MPI_DOUBLE, dst, 3, src, 3, cart_comm, MPI_STATUS_IGNORE);
    }
  }
}

std::vector<double> GatherResult(MPI_Comm cart_comm, int n, int bs, MPI_Datatype block_type, const int *sendcounts_ptr,
                                 const int *displs_ptr, const std::vector<double> &c_block) {
  int cart_rank = 0;
  MPI_Comm_rank(cart_comm, &cart_rank);

  std::vector<double> c_full;
  double *recv_ptr = nullptr;

  if (cart_rank == 0) {
    c_full.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    recv_ptr = c_full.data();
  }

  MPI_Gatherv(c_block.data(), bs * bs, MPI_DOUBLE, recv_ptr, sendcounts_ptr, displs_ptr, block_type, 0, cart_comm);
  return c_full;
}

std::vector<double> BroadcastFullToAll(std::vector<double> c_full, int world_rank) {
  int out_size = 0;
  if (world_rank == 0) {
    out_size = static_cast<int>(c_full.size());
  }
  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank != 0) {
    c_full.assign(static_cast<std::size_t>(out_size), 0.0);
  }

  if (out_size > 0) {
    MPI_Bcast(c_full.data(), out_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return c_full;
}

}  // namespace

MakoveevaSCannonAlgorithmMPI::MakoveevaSCannonAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool MakoveevaSCannonAlgorithmMPI::ValidationImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int n = BroadcastN(GetInput(), world_rank);

  int ok_input = 0;
  if (world_rank == 0) {
    const auto &a = std::get<0>(GetInput());
    const auto &b = std::get<1>(GetInput());
    ok_input = (GetOutput().empty() && CheckInputLocal(a, b, n)) ? 1 : 0;
  }
  MPI_Bcast(&ok_input, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int q = ChooseGridQ(world_size, n);
  const int ok_grid = (q > 0) ? 1 : 0;

  return (ok_input != 0) && (ok_grid != 0);
}

bool MakoveevaSCannonAlgorithmMPI::PreProcessingImpl() {
  if (!GetOutput().empty()) {
    GetOutput().clear();
  }
  return true;
}

bool MakoveevaSCannonAlgorithmMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int n = BroadcastN(GetInput(), world_rank);
  if (n <= 0) {
    return false;
  }

  const int q = ChooseGridQ(world_size, n);
  if (q <= 0) {
    return false;
  }

  const int active_p = q * q;
  bool is_active = false;
  MPI_Comm active_comm = MakeActiveComm(world_rank, active_p, &is_active);

  std::vector<double> c_full;

  if (is_active) {
    const int bs = n / q;
    const auto bs_sz = static_cast<std::size_t>(bs);

    MPI_Comm cart_comm = MakeCartComm(active_comm, q);
    const std::array<int, 2> coords = GetCoords(cart_comm);
    const int row = coords[0];
    const int col = coords[1];

    std::vector<double> a_block(bs_sz * bs_sz, 0.0);
    std::vector<double> b_block(bs_sz * bs_sz, 0.0);
    std::vector<double> c_block(bs_sz * bs_sz, 0.0);

    MPI_Datatype block_type = MakeBlockType(n, bs);

    std::vector<int> sendcounts;
    std::vector<int> displs;
    BuildScatterMeta(q, bs, n, active_p, &sendcounts, &displs, cart_comm);

    int cart_rank = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);
    const int *sendcounts_ptr = (cart_rank == 0) ? sendcounts.data() : nullptr;
    const int *displs_ptr = (cart_rank == 0) ? displs.data() : nullptr;

    const double *a_full_ptr = nullptr;
    const double *b_full_ptr = nullptr;
    if (world_rank == 0) {
      a_full_ptr = std::get<0>(GetInput()).data();
      b_full_ptr = std::get<1>(GetInput()).data();
    }

    ScatterBlocks(a_full_ptr, b_full_ptr, sendcounts_ptr, displs_ptr, block_type, bs, cart_comm, &a_block, &b_block);
    InitialShift(cart_comm, row, col, bs, &a_block, &b_block);
    CannonLoop(cart_comm, q, bs, &a_block, &b_block, &c_block);

    c_full = GatherResult(cart_comm, n, bs, block_type, sendcounts_ptr, displs_ptr, c_block);

    MPI_Type_free(&block_type);
    MPI_Comm_free(&cart_comm);
    MPI_Comm_free(&active_comm);
  }

  c_full = BroadcastFullToAll(std::move(c_full), world_rank);

  GetOutput() = std::move(c_full);
  return true;
}

bool MakoveevaSCannonAlgorithmMPI::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const int n = BroadcastN(GetInput(), world_rank);
  if (n <= 0) {
    return false;
  }

  const auto n_sz = static_cast<std::size_t>(n);
  return GetOutput().size() == (n_sz * n_sz);
}

}  // namespace makoveeva_s_cannon_algorithm
