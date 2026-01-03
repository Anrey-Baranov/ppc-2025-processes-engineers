#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsibareva_e_edge_select_sobel {

enum class ImageType : std::uint8_t { kTest1, kTest2, kTest3, kTest4, kTest5, kTest6, kTest7 };

using InType = std::tuple<std::vector<int>, int, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<ImageType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::tuple<std::vector<int>, int, int> ReadImageFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return {{}, 0, 0};
  }

  int height = 0;
  int width = 0;

  file >> height;
  file >> width;

  std::vector<int> pixels;
  pixels.reserve(static_cast<size_t>(height) * width);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int pixel = 0;
      if (!(file >> pixel)) {
        pixels = std::vector<int>();
      }
      pixels.push_back(pixel);
    }
  }

  file.close();
  return {pixels, height, width};
}

inline std::string GetDirectoryPath(const std::string &full_path) {
  std::string result = full_path;
  const std::string json_part = "settings.json";

  if (result.size() >= json_part.size() && result.ends_with(json_part)) {
    result.erase(result.size() - json_part.size(), json_part.size());
  }

  return result;
}

inline std::tuple<std::vector<int>, int, int, int> GenerateTestData(ImageType type) {
  std::string filename;

  switch (type) {
    case ImageType::kTest1:
      filename = "data/img1.txt";
      break;
    case ImageType::kTest2:
      filename = "data/img2.txt";
      break;
    case ImageType::kTest3:
      filename = "data/img3.txt";
      break;
    case ImageType::kTest4:
      filename = "data/img4.txt";
      break;
    case ImageType::kTest5:
      filename = "data/img5.txt";
      break;
    case ImageType::kTest6:
      filename = "data/img6.txt";
      break;
    case ImageType::kTest7:
      filename = "data/img7.txt";
      break;
    default:
      filename = "data/img1.txt";
  }

  std::string full_path = GetDirectoryPath(PPC_SETTINGS_tsibareva_e_edge_select_sobel) + filename;
  auto [pixels, height, width] = ReadImageFile(full_path);

  int threshold = 100;
  return {pixels, height, width, threshold};
}

inline std::vector<int> GenerateExpectedOutput(ImageType type) {
  std::string filename;

  switch (type) {
    case ImageType::kTest1:
      filename = "data/img1_res.txt";
      break;
    case ImageType::kTest2:
      filename = "data/img2_res.txt";
      break;
    case ImageType::kTest3:
      filename = "data/img3_res.txt";
      break;
    case ImageType::kTest4:
      filename = "data/img4_res.txt";
      break;
    case ImageType::kTest5:
      filename = "data/img5_res.txt";
      break;
    case ImageType::kTest6:
      filename = "data/img6_res.txt";
      break;
    case ImageType::kTest7:
      filename = "data/img7_res.txt";
      break;
    default:
      filename = "data/img1_res.txt";
  }

  std::string full_path = GetDirectoryPath(PPC_SETTINGS_tsibareva_e_edge_select_sobel) + filename;
  auto [pixels, height, width] = ReadImageFile(full_path);

  return pixels;
}

}  // namespace tsibareva_e_edge_select_sobel
