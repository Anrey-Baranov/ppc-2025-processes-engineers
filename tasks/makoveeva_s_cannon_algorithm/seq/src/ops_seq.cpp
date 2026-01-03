#include "makoveeva_s_cannon_algorithm/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "makoveeva_s_cannon_algorithm/common/include/common.hpp"

namespace makoveeva_s_cannon_algorithm {

namespace {

bool CheckMatrixSizes(const std::vector<double> &a, const std::vector<double> &b, int n) {
  if (n <= 0) {
    return false;
  }
  const auto n_sz = static_cast<std::size_t>(n);
  const auto expected = n_sz * n_sz;
  return (a.size() == expected) && (b.size() == expected);
}

void MultiplyDense(const std::vector<double> &a, const std::vector<double> &b, int n, std::vector<double> *c) {
  const auto n_sz = static_cast<std::size_t>(n);
  c->assign(n_sz * n_sz, 0.0);

  for (int i = 0; i < n; ++i) {
    const auto i_sz = static_cast<std::size_t>(i);
    for (int k = 0; k < n; ++k) {
      const auto k_sz = static_cast<std::size_t>(k);
      const double a_ik = a[(i_sz * n_sz) + k_sz];
      for (int j = 0; j < n; ++j) {
        const auto j_sz = static_cast<std::size_t>(j);
        (*c)[(i_sz * n_sz) + j_sz] += a_ik * b[(k_sz * n_sz) + j_sz];
      }
    }
  }
}

}  // namespace

MakoveevaSCannonAlgorithmSEQ::MakoveevaSCannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{};
}

bool MakoveevaSCannonAlgorithmSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  const int n = std::get<2>(input);

  if (!GetOutput().empty()) {
    return false;
  }
  return CheckMatrixSizes(a, b, n);
}

bool MakoveevaSCannonAlgorithmSEQ::PreProcessingImpl() {
  const int n = std::get<2>(GetInput());
  if (n <= 0) {
    return false;
  }
  const auto n_sz = static_cast<std::size_t>(n);
  GetOutput().assign(n_sz * n_sz, 0.0);
  return true;
}

bool MakoveevaSCannonAlgorithmSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  const int n = std::get<2>(input);

  if (!CheckMatrixSizes(a, b, n)) {
    return false;
  }

  std::vector<double> result;
  MultiplyDense(a, b, n, &result);
  GetOutput() = std::move(result);
  return true;
}

bool MakoveevaSCannonAlgorithmSEQ::PostProcessingImpl() {
  const int n = std::get<2>(GetInput());
  if (n <= 0) {
    return false;
  }
  const auto n_sz = static_cast<std::size_t>(n);
  return GetOutput().size() == (n_sz * n_sz);
}

}  // namespace makoveeva_s_cannon_algorithm
