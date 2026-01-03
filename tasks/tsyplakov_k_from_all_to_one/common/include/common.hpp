#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsyplakov_k_from_all_to_one {

template <typename T>
using InTypeT = std::tuple<std::vector<T>, int>;

template <typename T>
using OutTypeT = std::vector<T>;

using TestType = std::tuple<std::vector<int>, int, std::string>;
template <typename T>
using BaseTaskT = ppc::task::Task<InTypeT<T>, OutTypeT<T>>;

}  // namespace tsyplakov_k_from_all_to_one
