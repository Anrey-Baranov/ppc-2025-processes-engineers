#include "posternak_a_radix_merge_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "posternak_a_radix_merge_sort/common/include/common.hpp"

namespace posternak_a_radix_merge_sort {

PosternakARadixMergeSortMPI::PosternakARadixMergeSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PosternakARadixMergeSortMPI::ValidationImpl() {
  const std::vector<int> &input = GetInput();
  return !input.empty();
}

bool PosternakARadixMergeSortMPI::PreProcessingImpl() {
  return true;
}

std::vector<uint32_t> PosternakARadixMergeSortMPI::RadixSortLocal(std::vector<int> &data) {
  constexpr int kNumBytes = 4;
  constexpr uint32_t kByteMask = 0xFFU;

  std::vector<uint32_t> unsigned_data(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    unsigned_data[i] = static_cast<uint32_t>(data[i]) ^ 0x80000000U;
  }

  std::vector<uint32_t> buffer(data.size());

  for (int byte_index = 0; byte_index < kNumBytes; byte_index++) {
    std::vector<int> count(256, 0);
    for (uint32_t value : unsigned_data) {
      const auto current_byte = static_cast<uint8_t>((value >> (byte_index * 8)) & kByteMask);
      ++count[current_byte];
    }

    for (int i = 1; i < 256; ++i) {
      count[i] += count[i - 1];
    }

    for (int i = static_cast<int>(unsigned_data.size()) - 1; i >= 0; i--) {
      const auto current_byte = static_cast<uint8_t>((unsigned_data[i] >> (byte_index * 8)) & kByteMask);
      buffer[--count[current_byte]] = unsigned_data[i];
    }

    unsigned_data.swap(buffer);
  }

  return unsigned_data;
}

std::vector<int> PosternakARadixMergeSortMPI::ConvertToSigned(const std::vector<uint32_t> &unsigned_data) {
  std::vector<int> result(unsigned_data.size());
  for (size_t i = 0; i < unsigned_data.size(); i++) {
    result[i] = static_cast<int>(unsigned_data[i] ^ 0x80000000U);
  }
  return result;
}

void PosternakARadixMergeSortMPI::CalculateCountsAndOffsets(int input_len, int size, std::vector<int> &counts,
                                                            std::vector<int> &offset) {
  const int base = input_len / size;
  const int extra = input_len % size;

  counts.resize(size);
  offset.resize(size, 0);

  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < extra ? 1 : 0);
    if (i > 0) {
      offset[i] = offset[i - 1] + counts[i - 1];
    }
  }
}

void PosternakARadixMergeSortMPI::MergeSortedParts(std::vector<int> &result,
                                                   const std::vector<std::vector<int>> &sorted_proc_parts) {
  std::vector<int> tmp;
  for (size_t i = 1; i < sorted_proc_parts.size(); i++) {
    tmp.resize(result.size() + sorted_proc_parts[i].size());
    std::merge(result.begin(), result.end(), sorted_proc_parts[i].begin(), sorted_proc_parts[i].end(), tmp.begin());
    result.swap(tmp);
  }
}

bool PosternakARadixMergeSortMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int input_len = 0;
  std::vector<int> *input = nullptr;

  if (rank == 0) {
    input = &GetInput();
    input_len = static_cast<int>(input->size());
  }

  MPI_Bcast(&input_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts;
  std::vector<int> offset;
  CalculateCountsAndOffsets(input_len, size, counts, offset);

  const int local_size = counts[rank];
  std::vector<int> input_local(local_size);

  if (rank == 0) {
    MPI_Scatterv(input->data(), counts.data(), offset.data(), MPI_INT, input_local.data(), local_size, MPI_INT, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, counts.data(), offset.data(), MPI_INT, input_local.data(), local_size, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  std::vector<uint32_t> unsigned_sorted = RadixSortLocal(input_local);
  std::vector<int> local_sorted = ConvertToSigned(unsigned_sorted);

  std::vector<int> result;
  if (rank == 0) {
    std::vector<std::vector<int>> sorted_proc_parts;
    sorted_proc_parts.reserve(static_cast<size_t>(size));
    sorted_proc_parts.push_back(std::move(local_sorted));

    for (int proc = 1; proc < size; proc++) {
      std::vector<int> remote_sorted_proc_part(counts[proc]);
      MPI_Recv(remote_sorted_proc_part.data(), counts[proc], MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      sorted_proc_parts.push_back(std::move(remote_sorted_proc_part));
    }

    result = std::move(sorted_proc_parts[0]);
    MergeSortedParts(result, sorted_proc_parts);
  } else {
    MPI_Send(local_sorted.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  std::vector<int> output(input_len);
  if (rank == 0) {
    output = std::move(result);
  }

  MPI_Bcast(output.data(), input_len, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = std::move(output);

  return true;
}

bool PosternakARadixMergeSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace posternak_a_radix_merge_sort
