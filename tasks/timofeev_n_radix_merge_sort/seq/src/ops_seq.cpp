#include "timofeev_n_radix_merge_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>

#include "timofeev_n_radix_merge_sort/common/include/common.hpp"

namespace timofeev_n_radix_merge_sort {

TimofeevNRadixMergeSEQ::TimofeevNRadixMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

bool TimofeevNRadixMergeSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TimofeevNRadixMergeSEQ::PreProcessingImpl() {
  return true;
}

int TimofeevNRadixMergeSEQ::GetDigit(int num, int digit) {
  int abs_num = std::abs(num);

  // потому что без использования возведения в степень (как функции)
  for (int i = 0; i < digit; i++) {
    abs_num /= 10;
  }

  return abs_num % 10;
}

int TimofeevNRadixMergeSEQ::GetMaxDigits(const std::vector<int> &arr) {
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

void TimofeevNRadixMergeSEQ::SplitPosNeg(const std::vector<int> &arr, std::vector<int> &negative,
                                         std::vector<int> &positive) {
  for (int num : arr) {
    if (num < 0) {
      negative.push_back(-num);
    } else {
      positive.push_back(num);
    }
  }
}

void TimofeevNRadixMergeSEQ::RadixMergeBucketHelpingFunction(std::vector<int> &part, int digit) {
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

void TimofeevNRadixMergeSEQ::RadixMergeSort(std::vector<int> &part) {
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

bool TimofeevNRadixMergeSEQ::RunImpl() {
  GetOutput().resize(GetInput().size());
  for (size_t i = 0; i < GetInput().size(); i++) {
    GetOutput()[i] = GetInput()[i];
  }
  RadixMergeSort(GetOutput());
  return true;
}

bool TimofeevNRadixMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace timofeev_n_radix_merge_sort
