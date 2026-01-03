#include "dorogin_v_contrast_enhancement/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "dorogin_v_contrast_enhancement/common/include/common.hpp"

namespace dorogin_v_contrast_enhancement {

DoroginVContrastEnhancementMPI::DoroginVContrastEnhancementMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DoroginVContrastEnhancementMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int valid_flag = 0;
  if (world_rank_ == 0) {
    valid_flag = !GetInput().empty() ? 1 : 0;
  }

  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

bool DoroginVContrastEnhancementMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int global_size = 0;
  if (world_rank_ == 0) {
    image_ = GetInput();
    global_size = static_cast<int>(image_.size());
  }

  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);

  int base = global_size / world_size_;
  int rem = global_size % world_size_;
  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
  }
  for (int i = 1; i < world_size_; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  local_size_ = counts[world_rank_];
  local_image_.resize(local_size_);

  MPI_Scatterv(world_rank_ == 0 ? image_.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
               local_image_.data(), local_size_, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  result_.resize(global_size);

  return true;
}

bool DoroginVContrastEnhancementMPI::RunImpl() {
  unsigned char local_min = 255;
  unsigned char local_max = 0;

  if (local_size_ > 0) {
    local_min = *std::ranges::min_element(local_image_);
    local_max = *std::ranges::max_element(local_image_);
  }

  unsigned char global_min = 255;
  unsigned char global_max = 0;

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  std::vector<uint8_t> local_result(local_size_);

  if (global_min == global_max) {
    local_result = local_image_;
  } else {
    int range = global_max - global_min;
    for (int i = 0; i < local_size_; ++i) {
      local_result[i] = static_cast<uint8_t>(((local_image_[i] - global_min) * 255) / range);
    }
  }

  int global_size = static_cast<int>(result_.size());

  std::vector<int> counts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);

  int base = global_size / world_size_;
  int rem = global_size % world_size_;
  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
  }
  for (int i = 1; i < world_size_; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  MPI_Gatherv(local_result.data(), local_size_, MPI_UNSIGNED_CHAR, result_.data(), counts.data(), displs.data(),
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  MPI_Bcast(result_.data(), global_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool DoroginVContrastEnhancementMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace dorogin_v_contrast_enhancement
