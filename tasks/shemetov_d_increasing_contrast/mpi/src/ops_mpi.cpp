#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskMPI::IncreaseContrastTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool IncreaseContrastTaskMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool IncreaseContrastTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const size_t total_size = GetInput().size();

  std::vector<int> count(size, static_cast<int>(total_size / size));
  for (size_t i = 0; std::cmp_less(i, static_cast<int>(total_size % size)); ++i) {
    count[i]++;
  }

  std::vector<int> displacement(size, 0);
  for (int i = 1; i < size; ++i) {
    displacement[i] = displacement[i - 1] + count[i - 1];
  }

  std::vector<int> count_to_bytes(size);
  std::vector<int> displacement_to_bytes(size);

  for (int i = 0; i < size; ++i) {
    count_to_bytes[i] = count[i] * static_cast<int>(sizeof(Pixel));
    displacement_to_bytes[i] = displacement[i] * static_cast<int>(sizeof(Pixel));
  }

  std::vector<Pixel> local_input(count[rank]);
  std::vector<Pixel> local_output(count[rank]);

  MPI_Scatterv(GetInput().data(), count_to_bytes.data(), displacement_to_bytes.data(), MPI_BYTE, local_input.data(),
               count_to_bytes[rank], MPI_BYTE, 0, MPI_COMM_WORLD);

  constexpr float kFactor = 1.3F;

  for (size_t i = 0; i < local_input.size(); ++i) {
    const auto m_red = static_cast<float>(local_input[i].channel_red) * kFactor;
    const auto m_green = static_cast<float>(local_input[i].channel_red) * kFactor;
    const auto m_blue = static_cast<float>(local_input[i].channel_red) * kFactor;

    local_output[i].channel_red = static_cast<uint8_t>(std::clamp(m_red, 0.F, 255.F));
    local_output[i].channel_green = static_cast<uint8_t>(std::clamp(m_green, 0.F, 255.F));
    local_output[i].channel_blue = static_cast<uint8_t>(std::clamp(m_blue, 0.F, 255.F));
  }

  MPI_Gatherv(local_output.data(), count_to_bytes[rank], MPI_BYTE, GetOutput().data(), count_to_bytes.data(),
              displacement_to_bytes.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IncreaseContrastTaskMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace shemetov_d_increasing_contrast
