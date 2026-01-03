#include "zyazeva_s_graham_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "zyazeva_s_graham_scheme/common/include/common.hpp"

namespace zyazeva_s_graham_scheme {

namespace {

int Cross(const Point &origin, const Point &a, const Point &b) {
  return ((a.x - origin.x) * (b.y - origin.y)) - ((a.y - origin.y) * (b.x - origin.x));
}

void BuildConvexHullInPlace(std::vector<Point> &pts) {
  std::ranges::sort(pts.begin(), pts.end(),
                    [](const Point &a, const Point &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  std::vector<Point> hull;
  for (auto &p : pts) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  size_t lower_size = hull.size();
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    auto &p = pts[i];
    while (hull.size() > lower_size && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  pts = std::move(hull);
}

}  // namespace

ZyazevaSGrahamSchemeMPI::ZyazevaSGrahamSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZyazevaSGrahamSchemeMPI::ValidationImpl() {
  if (GetInput().size() < 3) {
    GetOutput().clear();
  }
  return true;
}

bool ZyazevaSGrahamSchemeMPI::PreProcessingImpl() {
  return true;
}

bool ZyazevaSGrahamSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 3) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }
  MPI_Datatype mpi_point = MPI_DATATYPE_NULL;
  MPI_Type_contiguous(2, MPI_INT, &mpi_point);
  MPI_Type_commit(&mpi_point);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int base = n / size;
  int rem = n % size;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
  }

  int local_n = sendcounts[rank];
  std::vector<Point> local_points(local_n);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), mpi_point,
               local_points.data(), local_n, mpi_point, 0, MPI_COMM_WORLD);

  if (local_points.size() >= 3) {
    BuildConvexHullInPlace(local_points);
  }

  int local_size = static_cast<int>(local_points.size());
  std::vector<int> recv_sizes(size);
  MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs_hull(size, 0);
  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      displs_hull[i] = total;
      total += recv_sizes[i];
    }
  }
  std::vector<Point> gathered_hulls(total);
  MPI_Gatherv(local_points.data(), local_size, mpi_point, gathered_hulls.data(), recv_sizes.data(), displs_hull.data(),
              mpi_point, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    BuildConvexHullInPlace(gathered_hulls);  // 3
    GetOutput() = std::move(gathered_hulls);
  }

  MPI_Type_free(&mpi_point);
  return true;
}

bool ZyazevaSGrahamSchemeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_graham_scheme
