#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector {

enum class MatrixType : std::uint8_t {
  kSingleConstant,  // 1x1
  kSingleRow,       // 1xN
  kSingleCol,       // Mx1
  kEmpty,           // 0x0
  kSquare,          // NxN
  kMoreRows,        // M>N
  kMoreCols,        // M<N
  kAllZeros,        // Все нули
  kPositive,        // Положительные
  kNegative,        // Отрицательные
  kMixedSigns       // Смешанные знаки
};

using InType = std::tuple<std::vector<int>, int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleConstant() {
  return {{5}, 1, 1, {2}};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {10};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleRow() {
  return {{1, 2, 3, 4, 5}, 1, 5, {1, 2, 3, 2, 1}};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {27};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleCol() {
  return {{1, 2, 3, 4}, 4, 1, {3}};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3, 6, 9, 12};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateEmpty() {
  return {{}, 0, 0, {}};
}
inline std::vector<int> GenerateEmptyExpected() {
  return {};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSquare() {
  return {{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3, {1, 1, 1}};
}
inline std::vector<int> GenerateSquareExpected() {
  return {1, 2, 3};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMoreRows() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2, {2, 1}};
}
inline std::vector<int> GenerateMoreRowsExpected() {
  return {4, 10, 16, 22, 28};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMoreCols() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 2, 5, {1, 0, 1, 0, 1}};
}
inline std::vector<int> GenerateMoreColsExpected() {
  return {9, 24};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateAllZeros() {
  std::vector<int> matrix(16, 0);
  return {matrix, 4, 4, {1, 2, 3, 4}};
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GeneratePositive() {
  return {{2, 2, 2, 3, 3, 3, 4, 4, 4}, 3, 3, {1, 1, 1}};
}
inline std::vector<int> GeneratePositiveExpected() {
  return {6, 9, 12};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateNegative() {
  return {{-1, -2, -3, -4, -5, -6, -7, -8, -9}, 3, 3, {-1, -1, -1}};
}
inline std::vector<int> GenerateNegativeExpected() {
  return {6, 15, 24};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMixedSigns() {
  return {{1, -2, 3, -4, -5, 6, -7, 8}, 2, 4, {1, -1, 1, -1}};
}
inline std::vector<int> GenerateMixedSignsExpected() {
  return {10, -26};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateTestData(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstant();
    case MatrixType::kSingleRow:
      return GenerateSingleRow();
    case MatrixType::kSingleCol:
      return GenerateSingleCol();
    case MatrixType::kEmpty:
      return GenerateEmpty();
    case MatrixType::kSquare:
      return GenerateSquare();
    case MatrixType::kMoreRows:
      return GenerateMoreRows();
    case MatrixType::kMoreCols:
      return GenerateMoreCols();
    case MatrixType::kAllZeros:
      return GenerateAllZeros();
    case MatrixType::kPositive:
      return GeneratePositive();
    case MatrixType::kNegative:
      return GenerateNegative();
    case MatrixType::kMixedSigns:
      return GenerateMixedSigns();
    default:
      return GenerateSingleConstant();
  }
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:
      return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:
      return GenerateSingleColExpected();
    case MatrixType::kEmpty:
      return GenerateEmptyExpected();
    case MatrixType::kSquare:
      return GenerateSquareExpected();
    case MatrixType::kMoreRows:
      return GenerateMoreRowsExpected();
    case MatrixType::kMoreCols:
      return GenerateMoreColsExpected();
    case MatrixType::kAllZeros:
      return GenerateAllZerosExpected();
    case MatrixType::kPositive:
      return GeneratePositiveExpected();
    case MatrixType::kNegative:
      return GenerateNegativeExpected();
    case MatrixType::kMixedSigns:
      return GenerateMixedSignsExpected();
    default:
      return GenerateSingleConstantExpected();
  }
}

}  // namespace tsibareva_e_ribbon_horizontal_matrix_mult_vector
