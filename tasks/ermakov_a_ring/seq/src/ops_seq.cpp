#include "ermakov_a_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ermakov_a_ring/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ermakov_a_ring {

namespace {

bool RunLocalLogic(ErmakovATestTaskSEQ *task, const int s, const int d) {
  auto &input_data = task->GetInput().data;
  if (!input_data.empty()) {
    for (size_t i = 0; i < input_data.size(); ++i) {
      input_data[i] = static_cast<int>(std::sqrt(std::abs(std::sin(static_cast<double>(i)))));
    }
  }

  const int v_size = std::max(s, d) + 1;
  const int fwd = (d - s + v_size) % v_size;
  const int bwd = (s - d + v_size) % v_size;

  bool go_fwd = (fwd <= bwd);

  std::vector<int> res;
  int curr = s;
  while (true) {
    res.push_back(curr);
    if (curr == d) {
      break;
    }
    if (go_fwd) {
      curr = (curr + 1) % v_size;
    } else {
      curr = (curr - 1 + v_size) % v_size;
    }
  }
  task->GetOutput() = res;
  return true;
}

bool RunMpiTopologyLogic(ErmakovATestTaskSEQ *task, const int s_idx, const int d_idx) {
  int w_rank = 0;
  int w_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &w_size);

  const int s = s_idx % w_size;
  const int d = d_idx % w_size;
  auto payload = task->GetInput().data;

  std::array<int, 1> dims = {w_size};
  std::array<int, 1> periods = {1};
  MPI_Comm ring_comm = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims.data(), periods.data(), 0, &ring_comm);

  int l_peer = 0;
  int r_peer = 0;
  MPI_Cart_shift(ring_comm, 0, 1, &l_peer, &r_peer);

  const int r_dist = (d - s + w_size) % w_size;
  const int l_dist = (s - d + w_size) % w_size;

  bool move_r = (r_dist <= l_dist);
  int nxt = l_peer;
  int prv = r_peer;
  int steps_total = l_dist;
  int my_dist = (s - w_rank + w_size) % w_size;

  if (move_r) {
    nxt = r_peer;
    prv = l_peer;
    steps_total = r_dist;
    my_dist = (w_rank - s + w_size) % w_size;
  }

  std::vector<int> path;
  if (w_rank == s) {
    path.push_back(s);
    if (s != d) {
      const int psz = static_cast<int>(path.size());
      MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, nxt, 100, ring_comm);
      MPI_Send(&psz, 1, MPI_INT, nxt, 10, ring_comm);
      MPI_Send(path.data(), psz, MPI_INT, nxt, 11, ring_comm);
    }
  }

  if (my_dist > 0 && my_dist <= steps_total) {
    int in_sz = 0;
    MPI_Recv(payload.data(), static_cast<int>(payload.size()), MPI_INT, prv, 100, ring_comm, MPI_STATUS_IGNORE);
    MPI_Recv(&in_sz, 1, MPI_INT, prv, 10, ring_comm, MPI_STATUS_IGNORE);
    path.resize(static_cast<size_t>(in_sz));
    MPI_Recv(path.data(), in_sz, MPI_INT, prv, 11, ring_comm, MPI_STATUS_IGNORE);
    path.push_back(w_rank);
    if (w_rank != d) {
      const int out_sz = static_cast<int>(path.size());
      MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, nxt, 100, ring_comm);
      MPI_Send(&out_sz, 1, MPI_INT, nxt, 10, ring_comm);
      MPI_Send(path.data(), out_sz, MPI_INT, nxt, 11, ring_comm);
    }
  }

  int total_len = static_cast<int>(path.size());
  MPI_Bcast(&total_len, 1, MPI_INT, d, ring_comm);
  if (w_rank != d) {
    path.resize(static_cast<size_t>(total_len));
  }
  MPI_Bcast(path.data(), total_len, MPI_INT, d, ring_comm);

  task->GetOutput() = path;
  MPI_Comm_free(&ring_comm);
  return true;
}

}  // namespace

ErmakovATestTaskSEQ::ErmakovATestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ErmakovATestTaskSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool ErmakovATestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool ErmakovATestTaskSEQ::RunImpl() {
  const int src = GetInput().source;
  const int dst = GetInput().dest;
  if (ppc::util::IsUnderMpirun()) {
    return RunMpiTopologyLogic(this, src, dst);
  }
  return RunLocalLogic(this, src, dst);
}

bool ErmakovATestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ermakov_a_ring
