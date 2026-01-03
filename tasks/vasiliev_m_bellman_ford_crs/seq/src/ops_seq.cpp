#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

#include <limits>
#include <vector>

#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

VasilievMBellmanFordCrsSEQ::VasilievMBellmanFordCrsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBellmanFordCrsSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.row_ptr.empty() || in.col_ind.empty() || in.vals.empty()) {
    return false;
  }

  if (in.col_ind.size() != in.vals.size()) {
    return false;
  }

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  if (vertices <= 0) {
    return false;
  }

  if (in.source < 0 || in.source >= vertices) {
    return false;
  }

  return true;
}

bool VasilievMBellmanFordCrsSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  const int inf = std::numeric_limits<int>::max();
  GetOutput().assign(vertices, inf);
  GetOutput()[in.source] = 0;

  return true;
}

bool VasilievMBellmanFordCrsSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &dist = GetOutput();

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  const int inf = std::numeric_limits<int>::max();

  for (int i = 0; i < vertices - 1; i++) {
    bool updated = false;

    for (int vertex = 0; vertex < vertices; vertex++) {
      if (dist[vertex] == inf) {
        continue;
      }

      for (int edge = in.row_ptr[vertex]; edge < in.row_ptr[vertex + 1]; edge++) {
        int v = in.col_ind[edge];
        int w = in.vals[edge];

        if (dist[v] > dist[vertex] + w) {
          dist[v] = dist[vertex] + w;
          updated = true;
        }
      }
    }

    if (!updated) {
      break;
    }
  }

  return true;
}

bool VasilievMBellmanFordCrsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bellman_ford_crs
