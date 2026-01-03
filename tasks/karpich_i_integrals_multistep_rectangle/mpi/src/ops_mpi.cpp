#include "karpich_i_integrals_multistep_rectangle/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "karpich_i_integrals_multistep_rectangle/common/include/common.hpp"
#include "task/include/task.hpp"

namespace karpich_i_integrals_multistep_rectangle {

namespace {

double IntegrandFunction(double x, double y, double z) {
  return (x * y * z) + (x * x) + (y * y) + (z * z);
}

double ComputeRectangleIntegralPartial(int n, int step, int start_i, int end_i) {
  double result = 0.0;
  int divisions = n * step;
  double h = 1.0 / divisions;

  for (int i = start_i; i < end_i; i++) {
    for (int j = 0; j < divisions; j++) {
      for (int k = 0; k < divisions; k++) {
        double x = (i + 0.5) * h;
        double y = (j + 0.5) * h;
        double z = (k + 0.5) * h;
        result += IntegrandFunction(x, y, z) * h * h * h;
      }
    }
  }

  return result;
}

}  // namespace

KarpichIIntegralsMultistepRectangleMPI::KarpichIIntegralsMultistepRectangleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIIntegralsMultistepRectangleMPI::ValidationImpl() {
  if (GetInput() <= 0) {
    return false;
  }
  if (GetOutput() != 0) {
    return false;
  }
  if (GetInput() > 1000) {
    return false;
  }
  if (GetStaticTypeOfTask() != ppc::task::TypeOfTask::kMPI) {
    return false;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0) {
    return false;
  }
  if (rank < 0 || rank >= size) {
    return false;
  }
  return true;
}

bool KarpichIIntegralsMultistepRectangleMPI::PreProcessingImpl() {
  if (GetInput() <= 0) {
    return false;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0) {
    return false;
  }
  GetOutput() = 0;
  if (GetOutput() != 0) {
    return false;
  }
  int divisions = GetInput() * 3;
  return divisions > 0;
}

bool KarpichIIntegralsMultistepRectangleMPI::RunImpl() {
  if (GetInput() == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double total_result = 0.0;

  for (int step = 1; step <= 3; step++) {
    int divisions = GetInput() * step;
    int rows_per_process = divisions / size;
    int remainder = divisions % size;

    int start_i = (rank * rows_per_process) + (rank < remainder ? rank : remainder);
    int end_i = start_i + rows_per_process + (rank < remainder ? 1 : 0);

    double local_result = ComputeRectangleIntegralPartial(GetInput(), step, start_i, end_i);

    if (rank == 0) {
      total_result += local_result;
      for (int proc = 1; proc < size; proc++) {
        double recv_result = 0.0;
        MPI_Recv(&recv_result, 1, MPI_DOUBLE, proc, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_result += recv_result;
      }
    } else {
      MPI_Send(&local_result, 1, MPI_DOUBLE, 0, step, MPI_COMM_WORLD);
    }
  }

  OutType final_output = 0;
  if (rank == 0) {
    total_result = total_result / 3.0;
    final_output = static_cast<OutType>(std::round(total_result));
  }

  MPI_Bcast(&final_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = final_output;

  return GetOutput() > 0;
}

bool KarpichIIntegralsMultistepRectangleMPI::PostProcessingImpl() {
  if (GetOutput() <= 0) {
    return false;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0) {
    return false;
  }
  if (GetInput() <= 0) {
    return false;
  }
  auto min_expected = static_cast<OutType>(1);
  if (GetOutput() < min_expected) {
    return false;
  }
  auto max_expected = static_cast<OutType>(GetInput() * GetInput() * GetInput() * 10);
  if (GetOutput() > max_expected) {
    return false;
  }
  if (rank == 0) {
    if (GetOutput() == GetInput() && GetInput() > 1) {
      return false;
    }
  }
  return true;
}

}  // namespace karpich_i_integrals_multistep_rectangle
