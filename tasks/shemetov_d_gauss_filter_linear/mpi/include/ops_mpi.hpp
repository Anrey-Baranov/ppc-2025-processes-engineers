#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit GaussFilterMPI(const InType &in);

 private:
  static Pixel ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel);
  static void ComputeLocalBlock(const InType &in, int start_row, std::vector<std::vector<Pixel>> &local_out);
  static std::vector<uint8_t> SendColumns(const std::vector<std::vector<Pixel>> &local_out, size_t column);
  static void RecieveColumns(std::vector<uint8_t> &recieve_columns, size_t column,
                             std::vector<std::vector<Pixel>> &out);
  static void GatherResult(const std::vector<std::vector<Pixel>> &local_out, const InType &in,
                           std::vector<std::vector<Pixel>> &out);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  inline static int rank = 0;
  inline static int size = 0;

  inline static int width = 0;
  inline static int height = 0;

  inline static int base_rows = 0;
  inline static int extra_rows = 0;
  inline static int local_rows = 0;
};

}  // namespace shemetov_d_gauss_filter_linear
