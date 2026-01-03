#include "Rastvorov_K_Number_of_character_alternations/seq/include/ops_seq.hpp"

#include <cstddef>

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

}  // namespace

RastvorovKNumberAfCharacterAlternationsSEQ::RastvorovKNumberAfCharacterAlternationsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RastvorovKNumberAfCharacterAlternationsSEQ::ValidationImpl() {
  return GetInput() >= 0;
}

bool RastvorovKNumberAfCharacterAlternationsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool RastvorovKNumberAfCharacterAlternationsSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    GetOutput() = 0;
    return true;
  }

  int prev = 0;
  int cnt = 0;

  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    const int s = Sign(GetElement(i));
    if (s == 0) {
      continue;
    }
    if (prev != 0 && s != prev) {
      ++cnt;
    }
    prev = s;
  }

  GetOutput() = cnt;
  return true;
}

bool RastvorovKNumberAfCharacterAlternationsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rastvorov_k_number_of_character_alternations
