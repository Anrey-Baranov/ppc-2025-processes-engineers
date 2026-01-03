#include "Rastvorov_K_Number_of_character_alternations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "Rastvorov_K_Number_of_character_alternations/common/include/common.hpp"

namespace rastvorov_k_number_of_character_alternations {

namespace {

inline int Sign(double x) {
  if (x > 0.0) {
    return 1;
  }
  if (x < 0.0) {
    return -1;
  }
  return 0;
}

inline double GetElement(std::size_t i) {
  if (i % 5 == 0) {
    return 0.0;
  }
  if (i % 2 == 0) {
    return 1.0;
  }
  return -1.0;
}

struct LocalInfo {
  int count{};
  int first_sign{};
  int last_sign{};
};

inline void ComputeRange(std::size_t rank, std::size_t size, std::size_t total, std::size_t *begin, std::size_t *end) {
  const std::size_t base = total / size;
  const std::size_t rem = total % size;

  if (rank < rem) {
    *begin = rank * (base + 1);
    *end = *begin + base + 1;
  } else {
    *begin = (rem * (base + 1)) + ((rank - rem) * base);
    *end = *begin + base;
  }
}

inline LocalInfo ProcessSegment(std::size_t begin, std::size_t end) {
  LocalInfo info{};

  for (std::size_t i = begin; i < end; ++i) {
    const int s = Sign(GetElement(i));
    if (s == 0) {
      continue;
    }

    if (info.first_sign == 0) {
      info.first_sign = s;
    }
    if (info.last_sign != 0 && info.last_sign != s) {
      ++info.count;
    }
    info.last_sign = s;
  }

  return info;
}

inline int CombineGlobal(const std::vector<int> &all_info) {
  int global_count = 0;
  int prev_sign = 0;
  const std::size_t size = all_info.size() / 3;

  for (std::size_t proc = 0; proc < size; ++proc) {
    const int lc = all_info[(3 * proc) + 0];
    const int fs = all_info[(3 * proc) + 1];
    const int ls = all_info[(3 * proc) + 2];

    if (fs != 0) {
      if (prev_sign != 0 && fs != prev_sign) {
        ++global_count;
      }
      if (ls != 0) {
        prev_sign = ls;
      }
    }

    global_count += lc;
  }

  return global_count;
}

}  // namespace

RastvorovKNumberAfCharacterAlternationsMPI::RastvorovKNumberAfCharacterAlternationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RastvorovKNumberAfCharacterAlternationsMPI::ValidationImpl() {
  return GetInput() >= 0;
}

bool RastvorovKNumberAfCharacterAlternationsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool RastvorovKNumberAfCharacterAlternationsMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const InType n = GetInput();
  if (n <= 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  const auto total = static_cast<std::size_t>(n);

  std::size_t begin = 0;
  std::size_t end = 0;
  ComputeRange(static_cast<std::size_t>(rank), static_cast<std::size_t>(size), total, &begin, &end);

  const LocalInfo local = ProcessSegment(begin, end);

  std::array<int, 3> local_info = {local.count, local.first_sign, local.last_sign};

  std::vector<int> all_info;
  if (rank == 0) {
    all_info.resize(static_cast<std::size_t>(size) * 3);
  }

  MPI_Gather(local_info.data(), 3, MPI_INT, rank == 0 ? all_info.data() : nullptr, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = CombineGlobal(all_info);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool RastvorovKNumberAfCharacterAlternationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rastvorov_k_number_of_character_alternations
