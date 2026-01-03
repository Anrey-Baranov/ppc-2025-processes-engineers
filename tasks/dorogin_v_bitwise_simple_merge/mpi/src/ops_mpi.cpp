#include "dorogin_v_bitwise_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dorogin_v_bitwise_simple_merge/common/include/common.hpp"

namespace dorogin_v_bitwise_simple_merge {

namespace {

uint64_t DoubleToSortableKey(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  if ((bits >> 63) != 0) {
    bits = ~bits;
  } else {
    bits ^= (static_cast<uint64_t>(1) << 63);
  }
  return bits;
}

double SortableKeyToDouble(uint64_t bits) {
  if ((bits >> 63) != 0) {
    bits ^= (static_cast<uint64_t>(1) << 63);
  } else {
    bits = ~bits;
  }
  double value = 0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

void RadixSort(std::vector<double> &data) {
  if (data.size() <= 1) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    keys[i] = DoubleToSortableKey(data[i]);
  }

  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(257, 0);

    for (const auto &key : keys) {
      size_t bucket = (key >> shift) & 0xFF;
      count[bucket + 1]++;
    }

    for (int i = 1; i <= 256; i++) {
      count[i] += count[i - 1];
    }

    for (const auto &key : keys) {
      size_t bucket = (key >> shift) & 0xFF;
      temp[count[bucket]++] = key;
    }

    std::swap(keys, temp);
  }

  for (size_t i = 0; i < data.size(); i++) {
    data[i] = SortableKeyToDouble(keys[i]);
  }
}

void SimpleMerge(const std::vector<double> &left, const std::vector<double> &right, std::vector<double> &result) {
  result.clear();
  result.reserve(left.size() + right.size());
  size_t i = 0;
  size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result.push_back(left[i]);
      i++;
    } else {
      result.push_back(right[j]);
      j++;
    }
  }
  while (i < left.size()) {
    result.push_back(left[i]);
    i++;
  }
  while (j < right.size()) {
    result.push_back(right[j]);
    j++;
  }
}

void DistributeData(const std::vector<double> &input_data, std::vector<double> &local_data,
                    const std::vector<int> &send_counts, const std::vector<int> &displs, int rank, int size) {
  int local_size = send_counts[rank];
  local_data.resize(local_size);

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      if (send_counts[proc] > 0) {
        MPI_Send(&input_data[displs[proc]], send_counts[proc], MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    }
    for (int i = 0; i < local_size; i++) {
      local_data[i] = input_data[i];
    }
  } else {
    if (local_size > 0) {
      MPI_Recv(local_data.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void GatherAndMerge(std::vector<double> &output_data, std::vector<double> &local_data,
                    const std::vector<int> &send_counts, int rank, int size) {
  if (rank == 0) {
    output_data = local_data;
    std::vector<double> temp_result;
    for (int proc = 1; proc < size; proc++) {
      if (send_counts[proc] > 0) {
        std::vector<double> recv_data(send_counts[proc]);
        MPI_Recv(recv_data.data(), send_counts[proc], MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        SimpleMerge(output_data, recv_data, temp_result);
        output_data.swap(temp_result);
      }
    }
  } else {
    int local_size = static_cast<int>(local_data.size());
    if (local_size > 0) {
      MPI_Send(local_data.data(), local_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
  }
}

}  // namespace

DoroginVBitwiseSimpleMergeMPI::DoroginVBitwiseSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DoroginVBitwiseSimpleMergeMPI::ValidationImpl() {
  return true;
}

bool DoroginVBitwiseSimpleMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    input_data_ = GetInput();
  }
  output_data_.clear();
  return true;
}

bool DoroginVBitwiseSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(input_data_.size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    output_data_.clear();
    return true;
  }

  int base_chunk = total_size / size;
  int remainder = total_size % size;

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    send_counts[i] = base_chunk + static_cast<int>(i < remainder);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + send_counts[i - 1]);
  }

  std::vector<double> local_data;
  DistributeData(input_data_, local_data, send_counts, displs, rank, size);

  RadixSort(local_data);

  GatherAndMerge(output_data_, local_data, send_counts, rank, size);

  output_data_.resize(static_cast<size_t>(total_size));
  if (total_size > 0) {
    MPI_Bcast(output_data_.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool DoroginVBitwiseSimpleMergeMPI::PostProcessingImpl() {
  GetOutput() = output_data_;
  return true;
}

}  // namespace dorogin_v_bitwise_simple_merge
