#include "balchunayte_z_shell_batcher/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "balchunayte_z_shell_batcher/common/include/common.hpp"

namespace balchunayte_z_shell_batcher {

namespace {

void ShellSort(std::vector<int> *vec) {
  auto &a = *vec;
  const std::size_t n = a.size();
  for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = a[i];
      std::size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
  }
}

struct Elem {
  int val{};
  bool pad{false};

  Elem(int value, bool padding) : val(value), pad(padding) {}
};

inline bool Greater(const Elem &lhs, const Elem &rhs) {
  if (lhs.pad != rhs.pad) {
    return lhs.pad && !rhs.pad;
  }
  return lhs.val > rhs.val;
}

inline void CompareExchange(std::vector<Elem> *arr, std::size_t i, std::size_t j) {
  if (Greater((*arr)[i], (*arr)[j])) {
    std::swap((*arr)[i], (*arr)[j]);
  }
}

std::size_t NextPow2(std::size_t x) {
  std::size_t p = 1;
  while (p < x) {
    p <<= 1U;
  }
  return p;
}

void OddEvenMergeStep(std::vector<Elem> *arr, std::size_t k, std::size_t j) {
  const std::size_t n = arr->size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t ixj = i ^ j;
    if (ixj > i) {
      if ((i & k) == 0) {
        CompareExchange(arr, i, ixj);
      } else {
        CompareExchange(arr, ixj, i);
      }
    }
  }
}

void OddEvenMergeNetwork(std::vector<Elem> *arr) {
  const std::size_t n = arr->size();
  if (n <= 1) {
    return;
  }

  for (std::size_t k = 2; k <= n; k <<= 1U) {
    for (std::size_t j = (k >> 1U); j > 0; j >>= 1U) {
      OddEvenMergeStep(arr, k, j);
    }
  }
}

std::vector<int> BatcherOddEvenMerge(const std::vector<int> &a, const std::vector<int> &b) {
  const std::size_t need = a.size() + b.size();
  const std::size_t half = NextPow2((a.size() > b.size()) ? a.size() : b.size());

  std::vector<Elem> buffer;
  buffer.reserve(2 * half);

  for (std::size_t i = 0; i < half; ++i) {
    if (i < a.size()) {
      buffer.emplace_back(a[i], false);
    } else {
      buffer.emplace_back(0, true);
    }
  }
  for (std::size_t i = 0; i < half; ++i) {
    if (i < b.size()) {
      buffer.emplace_back(b[i], false);
    } else {
      buffer.emplace_back(0, true);
    }
  }

  OddEvenMergeNetwork(&buffer);

  std::vector<int> out;
  out.reserve(need);
  for (const auto &elem : buffer) {
    if (!elem.pad) {
      out.push_back(elem.val);
    }
    if (out.size() == need) {
      break;
    }
  }
  return out;
}

}  // namespace

BalchunayteZShellBatcherSEQ::BalchunayteZShellBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BalchunayteZShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool BalchunayteZShellBatcherSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool BalchunayteZShellBatcherSEQ::RunImpl() {
  const std::size_t n = data_.size();
  if (n <= 1) {
    return true;
  }

  std::size_t blocks = 1;
  while ((blocks << 1U) <= n && (blocks << 1U) <= 16) {
    blocks <<= 1U;
  }

  std::vector<std::vector<int>> parts;
  parts.reserve(blocks);

  const std::size_t base = n / blocks;
  const std::size_t rem = n % blocks;

  std::size_t pos = 0;
  for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
    const std::size_t sz = base + (block_index < rem ? 1 : 0);
    std::vector<int> chunk;
    chunk.reserve(sz);
    for (std::size_t i = 0; i < sz; ++i) {
      chunk.push_back(data_[pos++]);
    }
    ShellSort(&chunk);
    parts.push_back(std::move(chunk));
  }

  while (parts.size() > 1) {
    std::vector<std::vector<int>> next;
    next.reserve(parts.size() / 2);
    for (std::size_t i = 0; i < parts.size(); i += 2) {
      next.push_back(BatcherOddEvenMerge(parts[i], parts[i + 1]));
    }
    parts = std::move(next);
  }

  data_ = std::move(parts[0]);
  return true;
}

bool BalchunayteZShellBatcherSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace balchunayte_z_shell_batcher
