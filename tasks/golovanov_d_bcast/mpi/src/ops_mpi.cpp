#include "golovanov_d_bcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "golovanov_d_bcast/common/include/common.hpp"

namespace golovanov_d_bcast {

GolovanovDBcastMPI::GolovanovDBcastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = false;
}

bool GolovanovDBcastMPI::ValidationImpl() {
  int n = std::get<1>(GetInput());
  auto size = static_cast<size_t>(n);
  return ((n > -1) && (std::get<2>(GetInput()).size() == size) && (std::get<3>(GetInput()).size() == size) &&
          (std::get<4>(GetInput()).size() == size) && !GetOutput());
}

bool GolovanovDBcastMPI::PreProcessingImpl() {
  int &index = std::get<0>(GetInput());
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (index >= world_size || index < 0) {
    index = 0;
  }
  return true;
}

bool GolovanovDBcastMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int main_proc = std::get<0>(GetInput());
  int n = 0;
  if (rank == main_proc) {
    n = std::get<1>(GetInput());
  }
  MyBcast(&n, 1, MPI_INT, main_proc, MPI_COMM_WORLD);
  std::vector<int> v_int(n);
  std::vector<float> v_float(n);
  std::vector<double> v_double(n);
  if (rank == main_proc) {
    v_int = std::get<2>(GetInput());
    v_float = std::get<3>(GetInput());
    v_double = std::get<4>(GetInput());
  }
  MyBcast(v_int.data(), n, MPI_INT, main_proc, MPI_COMM_WORLD);
  MyBcast(v_float.data(), n, MPI_FLOAT, main_proc, MPI_COMM_WORLD);
  MyBcast(v_double.data(), n, MPI_DOUBLE, main_proc, MPI_COMM_WORLD);

  GetOutput() = true;
  return true;
}

bool GolovanovDBcastMPI::PostProcessingImpl() {
  return true;
}

int GolovanovDBcastMPI::MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int real_rank = 0;
  MPI_Comm_rank(comm, &real_rank);
  int world_size = 0;
  MPI_Comm_size(comm, &world_size);
  int local_rank = (real_rank - root + world_size) % world_size;
  int rank_lvl = 0;
  // корень отправляет первому
  if (local_rank == 0) {
    if (world_size > 1) {
      rank_lvl = 1;
      int local_child = 1;
      int real_child = (root + local_child) % world_size;
      MPI_Send(buffer, count, datatype, real_child, 0, comm);
    }
  }
  // не-корень получает впервые
  else {
    rank_lvl = static_cast<int>(floor(log2(local_rank))) + 1;
    int parent_offset = static_cast<int>(pow(2, rank_lvl - 1));
    int local_parent = local_rank - parent_offset;
    int real_parent = (root + local_parent) % world_size;
    MPI_Recv(buffer, count, datatype, real_parent, 0, comm, MPI_STATUS_IGNORE);
  }
  // расслыка
  int local_child = local_rank + static_cast<int>(pow(2, rank_lvl));
  while (local_child < world_size) {
    int real_child = (root + local_child) % world_size;
    MPI_Send(buffer, count, datatype, real_child, 0, comm);
    rank_lvl++;
    local_child = local_rank + static_cast<int>(pow(2, rank_lvl));
  }
  return MPI_SUCCESS;
}

}  // namespace golovanov_d_bcast
