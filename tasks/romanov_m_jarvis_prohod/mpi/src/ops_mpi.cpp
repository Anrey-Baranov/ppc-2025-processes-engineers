#include "romanov_m_jarvis_prohod/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "romanov_m_jarvis_prohod/common/include/common.hpp"

namespace romanov_m_jarvis_prohod {

RomanovMJarvisProhodMPI::RomanovMJarvisProhodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RomanovMJarvisProhodMPI::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RomanovMJarvisProhodMPI::PreProcessingImpl() {
  return true;
}

namespace {

int LeftPoint(const std::vector<Point> &points) {
  int idx = 0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (points[i].x < points[idx].x || (points[i].x == points[idx].x && points[i].y < points[idx].y)) {
      idx = static_cast<int>(i);
    }
  }
  return idx;
}

int ChooseNextBoundaryPoint(const std::vector<Point> &points, int p) {
  const int n = static_cast<int>(points.size());
  int q = (p + 1) % n;

  for (int i = 0; i < n; ++i) {
    const int64_t cross = CrossCalculate(points[p], points[i], points[q]);
    if (cross > 0) {
      q = i;
    } else if (cross == 0) {
      if (SqDistance(points[p], points[i]) > SqDistance(points[p], points[q])) {
        q = i;
      }
    }
  }
  return q;
}

void BuildPointDatatype(MPI_Datatype *p_type) {
  MPI_Type_contiguous(2, MPI_INT, p_type);
  MPI_Type_commit(p_type);
}

void ComputeScatterLayout(int rank, int size, int n, std::vector<int> &counts, std::vector<int> &displs) {
  if (rank != 0) {
    return;
  }

  const int base = n / size;
  const int rem = n % size;

  for (int i = 0; i < size; ++i) {
    counts[i] = (i < rem) ? (base + 1) : base;
  }

  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

}  // namespace

std::vector<Point> RomanovMJarvisProhodMPI::JarvisMarch(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> hull;
  const int start = LeftPoint(points);

  int p = start;
  while (true) {
    hull.push_back(points[p]);
    p = ChooseNextBoundaryPoint(points, p);
    if (p == start) {
      break;
    }
  }

  return hull;
}

std::vector<Point> RomanovMJarvisProhodMPI::FinalHull(int rank, std::vector<Point> &all_hull_points) {
  if (rank != 0) {
    return {};
  }
  return JarvisMarch(std::move(all_hull_points));
}

bool RomanovMJarvisProhodMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 3) {
    if (rank == 0) {
      GetOutput() = GetInput();
    }
    return true;
  }

  MPI_Datatype p_type = MPI_DATATYPE_NULL;
  BuildPointDatatype(&p_type);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  ComputeScatterLayout(rank, size, n, counts, displs);

  const int l_size = (rank < (n % size)) ? ((n / size) + 1) : (n / size);
  std::vector<Point> local_points(static_cast<std::size_t>(l_size));

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, counts.data(), displs.data(), p_type, local_points.data(),
               l_size, p_type, 0, MPI_COMM_WORLD);

  std::vector<Point> local_hull = JarvisMarch(local_points);
  int local_size = static_cast<int>(local_hull.size());

  std::vector<int> recv_counts(size);
  std::vector<int> recv_displs(size);

  MPI_Gather(&local_size, 1, MPI_INT, rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      recv_displs[i] = total;
      total += recv_counts[i];
    }
  }

  std::vector<Point> all_points(static_cast<std::size_t>(total));
  MPI_Gatherv(local_hull.data(), local_size, p_type, rank == 0 ? all_points.data() : nullptr, recv_counts.data(),
              recv_displs.data(), p_type, 0, MPI_COMM_WORLD);

  std::vector<Point> final_hull = FinalHull(rank, all_points);
  int final_size = static_cast<int>(final_hull.size());

  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    final_hull.resize(static_cast<std::size_t>(final_size));
  }
  MPI_Bcast(final_hull.data(), final_size, p_type, 0, MPI_COMM_WORLD);

  GetOutput() = std::move(final_hull);

  MPI_Type_free(&p_type);
  return true;
}

bool RomanovMJarvisProhodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_jarvis_prohod
