#include "timofeev_n_radix_merge_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "timofeev_n_radix_merge_sort/common/include/common.hpp"

namespace timofeev_n_radix_merge_sort {

TimofeevNRadixMergeMPI::TimofeevNRadixMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

bool TimofeevNRadixMergeMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool TimofeevNRadixMergeMPI::PreProcessingImpl() {
  return true;
}

int TimofeevNRadixMergeMPI::GetDigit(int num, int digit) {
  int abs_num = std::abs(num);

  // потому что без использования возведения в степень (как функции)
  for (int i = 0; i < digit; i++) {
    abs_num /= 10;
  }

  return abs_num % 10;
}

int TimofeevNRadixMergeMPI::GetMaxDigits(const std::vector<int> &arr) {
  if (arr.empty()) {
    return 0;
  }

  // Находим максимальное по модулю число
  int max_abs = 0;
  for (int num : arr) {
    int abs_num = std::abs(num);
    max_abs = std::max(abs_num, max_abs);
  }

  int digits = 0;
  while (max_abs > 0) {
    digits++;
    max_abs /= 10;
  }
  return digits == 0 ? 1 : digits;  // минимум 1 разряд
}

void TimofeevNRadixMergeMPI::SplitPosNeg(const std::vector<int> &arr, std::vector<int> &negative,
                                         std::vector<int> &positive) {
  for (int num : arr) {
    if (num < 0) {
      negative.push_back(-num);
    } else {
      positive.push_back(num);
    }
  }
}

void TimofeevNRadixMergeMPI::RadixMergeBucketHelpingFunction(std::vector<int> &part, int digit) {
  std::vector<std::vector<int>> buckets(10);
  for (int num : part) {
    int d = GetDigit(num, digit);
    buckets[d].push_back(num);
  }

  // Собираем числа обратно в вектор
  part.clear();
  for (int i = 0; i < 10; i++) {
    for (int num : buckets[i]) {
      part.push_back(num);
    }
  }
}

void TimofeevNRadixMergeMPI::RadixMergeSort(std::vector<int> &part) {
  if (part.size() <= 1) {
    return;
  }

  // Разделяем на отрицательные и положительные числа
  std::vector<int> negative;
  std::vector<int> positive;
  SplitPosNeg(part, negative, positive);

  // Сортируем отрицательные числа (по модулю)
  if (!negative.empty()) {
    int max_digits_neg = GetMaxDigits(negative);

    for (int digit = 0; digit < max_digits_neg; digit++) {
      RadixMergeBucketHelpingFunction(negative, digit);
    }
  }

  // Сортируем положительные числа
  if (!positive.empty()) {
    int max_digits_pos = GetMaxDigits(positive);

    for (int digit = 0; digit < max_digits_pos; digit++) {
      RadixMergeBucketHelpingFunction(positive, digit);
    }
  }
  // А теперь сливаем всё в один вектор, причём негативные - задом наперёд,
  // позитивные - как было.
  // Вот так.
  if (!negative.empty()) {
    int j = 0;
    for (int &i : std::ranges::reverse_view(negative)) {
      part[j] = -i;
      j++;
    }
  }
  if (!positive.empty()) {
    for (size_t i = 0; i < positive.size(); i++) {
      part[i + negative.size()] = positive[i];
    }
  }
}

void TimofeevNRadixMergeMPI::SliyanieHelp(std::vector<std::vector<int>> &received, std::vector<int> &indexes,
                                          std::vector<int> &out, int &i) {
  int smales = INT_MAX;
  int index_smales = 0;
  for (size_t j = 0; j < received.size(); j++) {
    if (static_cast<size_t>(indexes[j]) < received[j].size() && received[j][indexes[j]] <= smales) {
      index_smales = static_cast<int>(j);
      smales = received[j][indexes[j]];
    }
  }
  out[i] = received[index_smales][indexes[index_smales]];
  indexes[index_smales]++;
}

void TimofeevNRadixMergeMPI::Sliyanie(std::vector<std::vector<int>> &received, std::vector<int> &out) {
  std::vector<int> indexes(received.size(), 0);
  for (int i = 0; static_cast<size_t>(i) < out.size(); i++) {
    SliyanieHelp(received, indexes, out, i);
  }
}

void TimofeevNRadixMergeMPI::HandleTwoProcesses(std::vector<int> &in, std::vector<int> &out) {
  int yes = 1;
  MPI_Send(&yes, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  size_t size_size = in.size();
  MPI_Send(&size_size, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD);
  MPI_Send(in.data(), static_cast<int>(size_size), MPI_INT, 1, 0, MPI_COMM_WORLD);
  out.resize(in.size());
  MPI_Recv(out.data(), static_cast<int>(size_size), MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  size_t concdlusion_size = out.size();
  MPI_Bcast(&concdlusion_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(out.data(), static_cast<int>(concdlusion_size), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

bool TimofeevNRadixMergeMPI::HandleZeroRank(std::vector<int> &in, std::vector<int> &out, int &size) {
  if (size == 2) {
    HandleTwoProcesses(in, out);
    return true;
  }
  int served_size = size - 1;
  size_t k =
      (in.size() / static_cast<size_t>(served_size)) + (in.size() % static_cast<size_t>(served_size) > 0 ? 1 : 0);
  std::vector<int> sizes_for_processes(served_size, 0);
  size_t to_sort_size = in.size();
  for (size_t i = 0; i < sizes_for_processes.size() - 1; i += k) {
    sizes_for_processes[i] = static_cast<int>((to_sort_size - k > 0 ? k : 0));
    to_sort_size -= static_cast<int>((to_sort_size > k ? k : 0));
  }
  sizes_for_processes[served_size - 1] = static_cast<int>(to_sort_size);
  for (int i = 0; i < served_size; i++) {
    int yes = 1;
    MPI_Send(&yes, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
  }
  for (int i = served_size; i < size - 1; i++) {
    int no = 0;
    MPI_Send(&no, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
  }
  int accumulated_size = 0;
  for (size_t i = 0; i < sizes_for_processes.size(); i++) {
    std::vector<int> buf_vec(in.begin() + accumulated_size, in.begin() + accumulated_size + sizes_for_processes[i]);
    size_t cur_size = buf_vec.size();
    MPI_Send(&cur_size, 1, MPI_UNSIGNED_LONG, static_cast<int>(i + 1), 0, MPI_COMM_WORLD);
    MPI_Send(buf_vec.data(), static_cast<int>(cur_size), MPI_INT, static_cast<int>(i + 1), 0, MPI_COMM_WORLD);
    accumulated_size += sizes_for_processes[i];
  }
  std::vector<std::vector<int>> received(sizes_for_processes.size(), std::vector<int>(1, 0));
  for (size_t i = 0; i < sizes_for_processes.size(); i++) {
    received[i].resize(sizes_for_processes[i]);
    MPI_Recv(received[i].data(), sizes_for_processes[i], MPI_INT, static_cast<int>(i + 1), 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  if (out.size() != in.size()) {
    out.resize(in.size());
  }
  for (size_t i = 0; i < in.size(); i++) {
    out[i] = in[i];
  }
  Sliyanie(received, out);
  size_t concdlusion_size = out.size();
  MPI_Bcast(&concdlusion_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(out.data(), static_cast<int>(concdlusion_size), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool TimofeevNRadixMergeMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    GetOutput().resize(GetInput().size());
    for (size_t i = 0; i < GetInput().size(); i++) {
      GetOutput()[i] = GetInput()[i];
    }
    RadixMergeSort(GetOutput());
    return true;
  }
  if (rank == 0) {
    return HandleZeroRank(GetInput(), GetOutput(), size);
  }
  int should_i_work = 0;
  MPI_Recv(&should_i_work, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (should_i_work == 0) {
    size_t concdlusion_size = 0;
    MPI_Bcast(&concdlusion_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    GetOutput().resize(concdlusion_size);
    MPI_Bcast(GetOutput().data(), static_cast<int>(concdlusion_size), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }
  size_t cur_size = 0;
  MPI_Recv(&cur_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> buf_vec(cur_size);
  MPI_Recv(buf_vec.data(), static_cast<int>(cur_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  RadixMergeSort(buf_vec);
  MPI_Send(buf_vec.data(), static_cast<int>(cur_size), MPI_INT, 0, 0, MPI_COMM_WORLD);
  size_t concdlusion_size = 0;
  MPI_Bcast(&concdlusion_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  GetOutput().resize(concdlusion_size);
  MPI_Bcast(GetOutput().data(), static_cast<int>(concdlusion_size), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool TimofeevNRadixMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace timofeev_n_radix_merge_sort
