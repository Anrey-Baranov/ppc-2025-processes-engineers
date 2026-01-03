#include "posternak_a_increase_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "posternak_a_increase_contrast/common/include/common.hpp"

namespace posternak_a_increase_contrast {

PosternakAIncreaseContrastMPI::PosternakAIncreaseContrastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool PosternakAIncreaseContrastMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool PosternakAIncreaseContrastMPI::PreProcessingImpl() {
  return true;
}

bool PosternakAIncreaseContrastMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int data_len = 0;
  if (rank == 0) {
    data_len = static_cast<int>(GetInput().size());
  }

  // Рассылаем процессам размер данных
  MPI_Bcast(&data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Раздаем локальные данные всем процессам
  std::vector<unsigned char> proc_part = ScatterInputData(rank, size, data_len);

  // Узнаем максимальное и минимальное значение пикселей и сообщаем об этом всем процессам
  unsigned char data_min = 0;
  unsigned char data_max = 0;
  FindGlobalMinMax(proc_part, &data_min, &data_max);

  // Преобразование пикселей процессами
  std::vector<unsigned char> local_output = ApplyContrast(proc_part, data_min, data_max);

  // Получаем данные о частях
  int local_size = data_len / size;
  int remainder = data_len % size;
  std::vector<int> counts(size);
  std::vector<int> step(size);
  int start = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    step[i] = start;
    start += counts[i];
  }

  // Рассылаем результат всем процессам
  GetOutput().resize(data_len);
  MPI_Allgatherv(local_output.data(), static_cast<int>(local_output.size()), MPI_UNSIGNED_CHAR, GetOutput().data(),
                 counts.data(), step.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

  return true;
}

std::vector<unsigned char> PosternakAIncreaseContrastMPI::ScatterInputData(int rank, int size, int data_len) {
  int local_size = data_len / size;
  int remainder = data_len % size;
  int my_size = local_size + (rank < remainder ? 1 : 0);

  std::vector<int> counts(size);
  std::vector<int> step(size);
  int start = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    step[i] = start;
    start += counts[i];
  }

  std::vector<unsigned char> proc_part(my_size);
  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), counts.data(), step.data(), MPI_UNSIGNED_CHAR, proc_part.data(), my_size,
                 MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR, proc_part.data(), my_size, MPI_UNSIGNED_CHAR, 0,
                 MPI_COMM_WORLD);
  }
  return proc_part;
}

void PosternakAIncreaseContrastMPI::FindGlobalMinMax(const std::vector<unsigned char> &proc_part,
                                                     unsigned char *data_min, unsigned char *data_max) {
  unsigned char local_min = 255;
  unsigned char local_max = 0;
  for (unsigned char pixel : proc_part) {
    local_min = std::min(local_min, pixel);
    local_max = std::max(local_max, pixel);
  }

  MPI_Allreduce(&local_min, data_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, data_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
}

std::vector<unsigned char> PosternakAIncreaseContrastMPI::ApplyContrast(const std::vector<unsigned char> &proc_part,
                                                                        unsigned char data_min,
                                                                        unsigned char data_max) {
  int my_size = static_cast<int>(proc_part.size());
  std::vector<unsigned char> local_output(my_size);

  if (data_min == data_max) {
    std::ranges::fill(local_output, 128);
  } else {
    const double scale = 255.0 / (data_max - data_min);
    for (int i = 0; i < my_size; ++i) {
      double scaled_value = (proc_part[i] - data_min) * scale;
      int new_pixel = static_cast<int>(std::lround(scaled_value));
      if (new_pixel < 0) {
        new_pixel = 0;
      } else if (new_pixel > 255) {
        new_pixel = 255;
      }
      local_output[i] = static_cast<unsigned char>(new_pixel);
    }
  }
  return local_output;
}

bool PosternakAIncreaseContrastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace posternak_a_increase_contrast
