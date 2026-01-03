#include "shkryleva_s_vec_min_val/seq/include/ops_seq.hpp"

#include <climits>
#include <cstddef>
#include <vector>

#include "shkryleva_s_vec_min_val/common/include/common.hpp"

namespace shkryleva_s_vec_min_val {

ShkrylevaSVecMinValSEQ::ShkrylevaSVecMinValSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrylevaSVecMinValSEQ::ValidationImpl() {
  return true;
}

bool ShkrylevaSVecMinValSEQ::PreProcessingImpl() {
  GetOutput() = INT_MAX;
  return true;
}

bool ShkrylevaSVecMinValSEQ::RunImpl() {
  if (GetInput().empty()) {
    // Для пустого вектора возвращаем INT_MAX (как в MPI версии)
    GetOutput() = INT_MAX;
    return true;
  }

  int min_val = GetInput()[0];
  for (size_t i = 1; i < GetInput().size(); i++) {
    if (GetInput()[i] < min_val) {  // NOLINT
      min_val = GetInput()[i];
    }
  }

  GetOutput() = min_val;
  return true;
}

bool ShkrylevaSVecMinValSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkryleva_s_vec_min_val
