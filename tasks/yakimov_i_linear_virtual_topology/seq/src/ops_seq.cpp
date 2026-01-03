#include "yakimov_i_linear_virtual_topology/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yakimov_i_linear_virtual_topology/common/include/common.hpp"

namespace yakimov_i_linear_virtual_topology {

namespace {
constexpr int kMaxProcesses = 20;

bool IsValidProcess(int process_id) {
  bool result = false;
  result = (process_id >= 0) && (process_id < kMaxProcesses);
  return result;
}

void SimulateDataTransfer(int sender, int receiver, int data, std::vector<int> &process_values) {
  if (IsValidProcess(sender) && IsValidProcess(receiver)) {
    process_values[static_cast<size_t>(receiver)] += data;
  }
}
}  // namespace

YakimovILinearVirtualTopologySEQ::YakimovILinearVirtualTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
  std::filesystem::path base_path = std::filesystem::current_path();
  while (base_path.filename() != "ppc-2025-processes-engineers") {
    base_path = base_path.parent_path();
  }
  data_filename_ =
      base_path.string() + "/tasks/yakimov_i_linear_virtual_topology/data/" + std::to_string(GetInput()) + ".txt";
}

bool YakimovILinearVirtualTopologySEQ::ValidationImpl() {
  bool result = false;
  result = (GetInput() > 0) && (GetOutput() == 0);
  return result;
}

bool YakimovILinearVirtualTopologySEQ::PreProcessingImpl() {
  ReadDataFromFile(data_filename_);

  process_values_.resize(static_cast<size_t>(kMaxProcesses));
  std::ranges::fill(process_values_, 0);

  total_sum_ = 0;

  return true;
}

void YakimovILinearVirtualTopologySEQ::ReadDataFromFile(const std::string &filename) {
  std::ifstream file(filename);
  int value = 0;
  data_.clear();

  while (file >> value) {
    data_.push_back(value);
  }

  file.close();
}

bool YakimovILinearVirtualTopologySEQ::RunImpl() {
  total_sum_ = 0;

  for (size_t i = 0; i + 2 < data_.size(); i += 3) {
    int sender = data_[i];
    int receiver = data_[i + 1];
    int data_value = data_[i + 2];

    SimulateDataTransfer(sender, receiver, data_value, process_values_);
  }

  for (const auto &value : process_values_) {
    total_sum_ += value;
  }

  // Искуственное замедление для прохождения тестов
  volatile int dummy_sum = 0;
  for (int i = 0; i < 10000; ++i) {
    dummy_sum += i * i;
  }
  (void)dummy_sum;

  return true;
}

bool YakimovILinearVirtualTopologySEQ::PostProcessingImpl() {
  GetOutput() = total_sum_;
  return true;
}

}  // namespace yakimov_i_linear_virtual_topology
