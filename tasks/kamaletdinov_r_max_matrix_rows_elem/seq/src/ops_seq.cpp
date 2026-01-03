#include "kamaletdinov_r_max_matrix_rows_elem/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kamaletdinov_r_max_matrix_rows_elem/common/include/common.hpp"

namespace kamaletdinov_r_max_matrix_rows_elem {
KamaletdinovRMaxMatrixRowsElemSEQ::KamaletdinovRMaxMatrixRowsElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::ValidationImpl() {
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());
  std::vector<int> &val = std::get<2>(GetInput());
  valid_ = (n > 0) && (m > 0) && (val.size() == (n * m));
  return valid_;
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::PreProcessingImpl() {
  if (valid_) {
    std::size_t m = std::get<0>(GetInput());
    std::size_t n = std::get<1>(GetInput());
    std::vector<int> &val = std::get<2>(GetInput());
    t_matrix_ = std::vector<int>(n * m);
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        t_matrix_[(j * m) + i] = val[(i * n) + j];
      }
    }
    return true;
  }
  return false;
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::RunImpl() {
  if (!valid_) {
    return false;
  }
  std::size_t m = std::get<0>(GetInput());
  std::size_t n = std::get<1>(GetInput());

  // debug
  //  std::string deb = "\n\n----\n";
  //  for(std::size_t i = 0; i < n; i++) {
  //    for(std::size_t j = 0; j < m; j++) {
  //      deb += std::to_string(t_matrix_[i*m + j]) + " ";
  //    }
  //    deb += "\n";
  //  }
  //  std::cout << deb;

  std::vector<int> max_rows_elem(n);
  for (std::size_t i = 0; i < n; i++) {
    max_rows_elem[i] = t_matrix_[(i * m)];
    for (std::size_t j = 1; j < m; j++) {
      max_rows_elem[i] = std::max(max_rows_elem[i], t_matrix_[(i * m) + j]);
    }
  }

  // debug output
  //  std::cout << "seq" << ":";
  //  for(std::size_t i = 0; i < n; i++) {
  //    std::cout << max_rows_elem[i] << " ";
  //  }
  //  std::cout << std::endl;

  GetOutput() = max_rows_elem;
  return true;
}

bool KamaletdinovRMaxMatrixRowsElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamaletdinov_r_max_matrix_rows_elem
