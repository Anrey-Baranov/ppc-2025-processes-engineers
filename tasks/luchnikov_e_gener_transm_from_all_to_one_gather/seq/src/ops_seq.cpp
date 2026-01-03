#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

namespace {
size_t GetTypeSizeSeq(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}
}  // namespace

LuchnikovETransmFrAllToOneGatherSEQ::LuchnikovETransmFrAllToOneGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LuchnikovETransmFrAllToOneGatherSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.data.empty()) {
    return false;
  }
  if (input.count <= 0) {
    return false;
  }

  if (input.root < 0) {
    return false;
  }

  size_t type_size = GetTypeSizeSeq(input.datatype);

  if (type_size == 0) {
    return false;
  }

  if (input.data.size() != static_cast<size_t>(input.count) * type_size) {
    return false;
  }

  return true;
}

bool LuchnikovETransmFrAllToOneGatherSEQ::PreProcessingImpl() {
  return true;
}

bool LuchnikovETransmFrAllToOneGatherSEQ::RunImpl() {
  const auto &input = GetInput();

  GetOutput() = input.data;

  return true;
}

bool LuchnikovETransmFrAllToOneGatherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
