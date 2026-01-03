#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_matrix_column_max {

enum class MatrixType : std::uint8_t {
  kSingleConstant,    // 1x1 константная
  kSingleRow,         // 1x10 одна строка
  kSingleCol,         // 3x1 один столбец
  kAllZeros,          // 5x5 все нули
  kConstant,          // 5x5 константная
  kMaxFirst,          // 6x4 максимум в первой строке
  kMaxLast,           // 6x4 максимум в последней строке
  kMaxMiddle,         // 6x4 максимум в середине
  kAscending,         // 8x8 возрастающая
  kDescending,        // 8x8 убывающая
  kDiagonalDominant,  // 8x8 диагонально доминантная
  kSparse,            // 8x8 разреженная
  kNegative,          // 8x8 отрицательная
  kSquareSmall,       // 2x2 маленькая квадратная
  kVertical,          // 10x4 вертикальная
  kHorizontal,        // 5x10 горизонтальная
  kCheckerboard,      // 7x7 шахматная
  kEmpty,             // Пустая матрица
  kZeroColumns,       // Матрица с нулевыми столбцами
  kNonRectangular     // Непрямоугольная матрица
};

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::vector<std::vector<int>> GenerateSingleConstantMatrix() {
  return {{30}};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {30};
}

inline std::vector<std::vector<int>> GenerateSingleRowMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
}

inline std::vector<std::vector<int>> GenerateSingleColMatrix() {
  return {{1}, {2}, {3}};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3};
}

inline std::vector<std::vector<int>> GenerateAllZerosMatrix() {
  return {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0, 0};
}

inline std::vector<std::vector<int>> GenerateConstantMatrix() {
  return {{30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}};
}
inline std::vector<int> GenerateConstantExpected() {
  return {30, 30, 30, 30, 30};
}

inline std::vector<std::vector<int>> GenerateMaxFirstMatrix() {
  return {{1000, 1001, 1002, 1003}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}};
}
inline std::vector<int> GenerateMaxFirstExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxLastMatrix() {
  return {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}, {1000, 1001, 1002, 1003}};
}
inline std::vector<int> GenerateMaxLastExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxMiddleMatrix() {
  return {{1, 2, 3, 4}, {2, 3, 4, 5}, {1000, 1001, 1002, 1003}, {4, 5, 6, 7}, {5, 6, 7, 8}, {6, 7, 8, 9}};
}
inline std::vector<int> GenerateMaxMiddleExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateAscendingMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8},         {9, 10, 11, 12, 13, 14, 15, 16},  {17, 18, 19, 20, 21, 22, 23, 24},
          {25, 26, 27, 28, 29, 30, 31, 32}, {33, 34, 35, 36, 37, 38, 39, 40}, {41, 42, 43, 44, 45, 46, 47, 48},
          {49, 50, 51, 52, 53, 54, 55, 56}, {57, 58, 59, 60, 61, 62, 63, 64}};
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}

inline std::vector<std::vector<int>> GenerateDescendingMatrix() {
  return {{64, 63, 62, 61, 60, 59, 58, 57}, {56, 55, 54, 53, 52, 51, 50, 49}, {48, 47, 46, 45, 44, 43, 42, 41},
          {40, 39, 38, 37, 36, 35, 34, 33}, {32, 31, 30, 29, 28, 27, 26, 25}, {24, 23, 22, 21, 20, 19, 18, 17},
          {16, 15, 14, 13, 12, 11, 10, 9},  {8, 7, 6, 5, 4, 3, 2, 1}};
}
inline std::vector<int> GenerateDescendingExpected() {
  return {64, 63, 62, 61, 60, 59, 58, 57};
}

inline std::vector<std::vector<int>> GenerateDiagonalDominantMatrix() {
  return {{1000, 1, 2, 3, 4, 5, 6, 7},    {1, 1100, 3, 4, 5, 6, 7, 8},    {2, 3, 1200, 5, 6, 7, 8, 9},
          {3, 4, 5, 1300, 7, 8, 9, 10},   {4, 5, 6, 7, 1400, 9, 10, 11},  {5, 6, 7, 8, 9, 1500, 11, 12},
          {6, 7, 8, 9, 10, 11, 1600, 14}, {7, 8, 9, 10, 11, 12, 14, 1700}};
}
inline std::vector<int> GenerateDiagonalDominantExpected() {
  return {1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700};
}

inline std::vector<std::vector<int>> GenerateSparseMatrix() {
  return {{8, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 61, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 72, 0},
          {0, 0, 0, 0, 0, 0, 0, 0}, {0, 9, 0, 0, 0, 0, 0, 0},  {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
}
inline std::vector<int> GenerateSparseExpected() {
  return {8, 9, 0, 61, 0, 0, 72, 0};
}

inline std::vector<std::vector<int>> GenerateNegativeMatrix() {
  return {{-30, -31, -32, -33, -34, -35, -36, -37}, {-40, -41, -42, -43, -44, -45, -46, -47},
          {-50, -51, -52, -53, -54, -55, -56, -57}, {-60, -61, -62, -63, -64, -65, -66, -67},
          {-70, -71, -72, -73, -74, -75, -76, -77}, {-80, -81, -82, -83, -84, -85, -86, -87},
          {-90, -91, -92, -93, -94, -95, -96, -97}, {-100, -101, -102, -103, -104, -105, -106, -107}};
}
inline std::vector<int> GenerateNegativeExpected() {
  return {-30, -31, -32, -33, -34, -35, -36, -37};
}

inline std::vector<std::vector<int>> GenerateSquareSmallMatrix() {
  return {{1, 2}, {3, 4}};
}
inline std::vector<int> GenerateSquareSmallExpected() {
  return {3, 4};
}

inline std::vector<std::vector<int>> GenerateVerticalMatrix() {
  return {{1, 2, 3, 4},     {5, 6, 7, 8},     {9, 10, 11, 12},  {13, 14, 15, 16}, {17, 18, 19, 20},
          {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}, {37, 38, 39, 40}};
}
inline std::vector<int> GenerateVerticalExpected() {
  return {37, 38, 39, 40};
}

inline std::vector<std::vector<int>> GenerateHorizontalMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
          {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
          {21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
          {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
          {41, 42, 43, 44, 45, 46, 47, 48, 49, 50}};
}
inline std::vector<int> GenerateHorizontalExpected() {
  return {41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
}

inline std::vector<std::vector<int>> GenerateCheckerboardMatrix() {
  return {{1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1},
          {1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1}};
}
inline std::vector<int> GenerateCheckerboardExpected() {
  return {1, 1, 1, 1, 1, 1, 1};
}

inline std::vector<std::vector<int>> GenerateEmptyMatrix() {
  return {};
}
inline std::vector<std::vector<int>> GenerateZeroColumnsMatrix() {
  return std::vector<std::vector<int>>(5, std::vector<int>());
}
inline std::vector<std::vector<int>> GenerateNonRectangularMatrix() {
  return {{1, 2, 3}, {4, 5}, {6, 7, 8}};
}
inline std::vector<int> GenerateEmptyExpected() {
  return {};
}

inline std::vector<std::vector<int>> GenerateMatrixFunc(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantMatrix();
    case MatrixType::kSingleRow:
      return GenerateSingleRowMatrix();
    case MatrixType::kSingleCol:
      return GenerateSingleColMatrix();
    case MatrixType::kAllZeros:
      return GenerateAllZerosMatrix();
    case MatrixType::kConstant:
      return GenerateConstantMatrix();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstMatrix();
    case MatrixType::kMaxLast:
      return GenerateMaxLastMatrix();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleMatrix();
    case MatrixType::kAscending:
      return GenerateAscendingMatrix();
    case MatrixType::kDescending:
      return GenerateDescendingMatrix();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantMatrix();
    case MatrixType::kSparse:
      return GenerateSparseMatrix();
    case MatrixType::kNegative:
      return GenerateNegativeMatrix();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallMatrix();
    case MatrixType::kVertical:
      return GenerateVerticalMatrix();
    case MatrixType::kHorizontal:
      return GenerateHorizontalMatrix();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardMatrix();
    case MatrixType::kEmpty:
      return GenerateEmptyMatrix();
    case MatrixType::kZeroColumns:
      return GenerateZeroColumnsMatrix();
    case MatrixType::kNonRectangular:
      return GenerateNonRectangularMatrix();
  }
  return GenerateSingleConstantMatrix();
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:
      return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:
      return GenerateSingleColExpected();
    case MatrixType::kAllZeros:
      return GenerateAllZerosExpected();
    case MatrixType::kConstant:
      return GenerateConstantExpected();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstExpected();
    case MatrixType::kMaxLast:
      return GenerateMaxLastExpected();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleExpected();
    case MatrixType::kAscending:
      return GenerateAscendingExpected();
    case MatrixType::kDescending:
      return GenerateDescendingExpected();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantExpected();
    case MatrixType::kSparse:
      return GenerateSparseExpected();
    case MatrixType::kNegative:
      return GenerateNegativeExpected();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallExpected();
    case MatrixType::kVertical:
      return GenerateVerticalExpected();
    case MatrixType::kHorizontal:
      return GenerateHorizontalExpected();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardExpected();
    case MatrixType::kEmpty:
    case MatrixType::kZeroColumns:
    case MatrixType::kNonRectangular:
      return GenerateEmptyExpected();
  }
  return GenerateSingleConstantExpected();
}

}  // namespace tsibareva_e_matrix_column_max
