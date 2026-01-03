#include "balchunayte_z_reduce/seq/include/ops_seq.hpp"

#include <numeric>

#include "balchunayte_z_reduce/common/include/common.hpp"

namespace balchunayte_z_reduce {

BalchunayteZReduceSEQ::BalchunayteZReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool BalchunayteZReduceSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.data.empty()) {
    return false;
  }
  if (in.root < 0) {
    return false;
  }
  return true;
}

bool BalchunayteZReduceSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool BalchunayteZReduceSEQ::RunImpl() {
  const auto &data = GetInput().data;
  GetOutput() = std::accumulate(data.begin(), data.end(), 0.0);
  return true;
}

bool BalchunayteZReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace balchunayte_z_reduce
