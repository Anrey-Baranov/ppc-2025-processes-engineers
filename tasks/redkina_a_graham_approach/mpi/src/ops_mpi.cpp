#include "redkina_a_graham_approach/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"

namespace redkina_a_graham_approach {

RedkinaAGrahamApproachMPI::RedkinaAGrahamApproachMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaAGrahamApproachMPI::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RedkinaAGrahamApproachMPI::PreProcessingImpl() {
  return true;
}

namespace {

inline bool ComparePolarAngle(const Point &pivot, const Point &a, const Point &b) noexcept {
  const int cross = CalcCross(pivot, a, b);
  if (cross == 0) {
    return CalcDistSq(pivot, a) < CalcDistSq(pivot, b);
  }
  return cross > 0;
}

void CreateMpiPointType(MPI_Datatype *p_type) {
  MPI_Type_contiguous(2, MPI_INT, p_type);
  MPI_Type_commit(p_type);
}

void ScatterPointData(int rank, const std::vector<Point> &a_points, std::vector<Point> &l_points,
                      const std::vector<int> &counts, const std::vector<int> &displs, MPI_Datatype p_type) {
  MPI_Scatterv(rank == 0 ? a_points.data() : nullptr, rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, p_type, l_points.data(), static_cast<int>(l_points.size()), p_type,
               0, MPI_COMM_WORLD);
}

void GatherLocalHulls(int rank, const std::vector<Point> &l_hull, std::vector<Point> &a_hull_points,
                      const std::vector<int> &r_counts, const std::vector<int> &r_displs, MPI_Datatype p_type) {
  MPI_Gatherv(l_hull.data(), static_cast<int>(l_hull.size()), p_type, rank == 0 ? a_hull_points.data() : nullptr,
              rank == 0 ? r_counts.data() : nullptr, rank == 0 ? r_displs.data() : nullptr, p_type, 0, MPI_COMM_WORLD);
}

void BroadcastHull(int rank, std::vector<Point> &f_hull, MPI_Datatype p_type) {
  int f_size = static_cast<int>(f_hull.size());
  MPI_Bcast(&f_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    f_hull.resize(f_size);
  }

  MPI_Bcast(f_hull.data(), f_size, p_type, 0, MPI_COMM_WORLD);
}

void InitCountsAndDispls(int rank, int size, int n, std::vector<int> &counts, std::vector<int> &displs) {
  if (rank == 0) {
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
}

}  // namespace

std::vector<Point> RedkinaAGrahamApproachMPI::GrahamScan(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  const Point pivot = FindPivotPoint(points);
  std::erase_if(points, [&pivot](const Point &p) { return ArePointsEqual(p, pivot); });

  std::ranges::sort(points, [&pivot](const Point &a, const Point &b) { return ComparePolarAngle(pivot, a, b); });

  std::vector<Point> hull;
  hull.reserve(points.size() + 1);
  hull.push_back(pivot);
  hull.push_back(points[0]);
  hull.push_back(points[1]);

  for (std::size_t i = 2; i < points.size(); ++i) {
    while (hull.size() >= 2 && CalcCross(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  return hull;
}

std::vector<Point> RedkinaAGrahamApproachMPI::ComputeFinalHull(int rank, std::vector<Point> &all_hull_points) {
  if (rank != 0) {
    return {};
  }

  if (all_hull_points.size() >= 3) {
    return GrahamScan(std::move(all_hull_points));
  }

  return std::move(all_hull_points);
}

bool RedkinaAGrahamApproachMPI::RunImpl() {
  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<Point> a_points;
  int n = 0;

  if (rank == 0) {
    a_points = GetInput();
    n = static_cast<int>(a_points.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 3) {
    GetOutput() = (rank == 0 ? a_points : std::vector<Point>{});
    return true;
  }

  MPI_Datatype p_type = MPI_DATATYPE_NULL;
  CreateMpiPointType(&p_type);

  const int base = n / size;
  const int rem = n % size;
  const int l_size = (rank < rem) ? (base + 1) : base;

  std::vector<Point> l_points(l_size);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  InitCountsAndDispls(rank, size, n, counts, displs);

  ScatterPointData(rank, a_points, l_points, counts, displs, p_type);

  std::vector<Point> l_hull = (l_size >= 3) ? GrahamScan(std::move(l_points)) : std::move(l_points);

  int l_count = static_cast<int>(l_hull.size());

  std::vector<int> r_counts(size);
  std::vector<int> r_displs(size);

  MPI_Gather(&l_count, 1, MPI_INT, rank == 0 ? r_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      r_displs[i] = (i == 0) ? 0 : r_displs[i - 1] + r_counts[i - 1];
      total += r_counts[i];
    }
  }

  std::vector<Point> a_hull_points(total);
  GatherLocalHulls(rank, l_hull, a_hull_points, r_counts, r_displs, p_type);

  std::vector<Point> f_hull = ComputeFinalHull(rank, a_hull_points);

  BroadcastHull(rank, f_hull, p_type);

  GetOutput() = std::move(f_hull);

  MPI_Type_free(&p_type);
  return true;
}

bool RedkinaAGrahamApproachMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_graham_approach
