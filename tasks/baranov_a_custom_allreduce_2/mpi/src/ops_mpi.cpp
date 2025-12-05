#include "baranov_a_custom_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <variant>

#include "baranov_a_custom_allreduce/common/include/common.hpp"

namespace baranov_a_custom_allreduce {

void BaranovACustomAllreduceMPI::TreeBroadcast(void *buffer, int count, MPI_Datatype datatype, MPI_Comm comm,
                                               int root) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (count == 0) {
    return;
  }
  if (rank == root) {
    for (int i = 0; i < size; i++) {
      if (i != root) {
        MPI_Send(buffer, count, datatype, i, 0, comm);
      }
    }
  } else {
    MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
  }
}

void BaranovACustomAllreduceMPI::TreeReduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                            MPI_Comm comm, int root) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (count == 0) {
    return;
  }

  int type_size;
  MPI_Type_size(datatype, &type_size);

  std::memcpy(recvbuf, sendbuf, count * type_size);
  if (rank == root) {
    for (int i = 0; i < size; i++) {
      if (i != root) {
        void *temp_buf = malloc(count * type_size);
        if (!temp_buf) {
          throw std::runtime_error("Memory allocation failed");
        }

        MPI_Recv(temp_buf, count, datatype, i, 0, comm, MPI_STATUS_IGNORE);
        PerformOperation(temp_buf, recvbuf, count, datatype, op);
        free(temp_buf);
      }
    }
  } else {
    MPI_Send(recvbuf, count, datatype, root, 0, comm);
    std::memset(recvbuf, 0, count * type_size);
  }
}

void BaranovACustomAllreduceMPI::PerformOperation(void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype,
                                                  MPI_Op op) {
  if (op != MPI_SUM) {
    throw std::runtime_error("Only MPI_SUM operation is supported");
  }

  if (datatype == MPI_INT) {
    int *in = static_cast<int *>(inbuf);
    int *inout = static_cast<int *>(inoutbuf);
    for (int i = 0; i < count; i++) {
      inout[i] += in[i];
    }
  } else if (datatype == MPI_FLOAT) {
    float *in = static_cast<float *>(inbuf);
    float *inout = static_cast<float *>(inoutbuf);
    for (int i = 0; i < count; i++) {
      inout[i] += in[i];
    }
  } else if (datatype == MPI_DOUBLE) {
    double *in = static_cast<double *>(inbuf);
    double *inout = static_cast<double *>(inoutbuf);
    for (int i = 0; i < count; i++) {
      inout[i] += in[i];
    }
  } else {
    throw std::runtime_error("Unsupported datatype");
  }
}

void BaranovACustomAllreduceMPI::CustomAllreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                                                 MPI_Op op, MPI_Comm comm, int root) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (count == 0) {
    return;
  }

  int type_size;
  MPI_Type_size(datatype, &type_size);
  void *temp_buf = malloc(count * type_size);
  if (!temp_buf) {
    throw std::runtime_error("Memory allocation failed");
  }

  std::memcpy(temp_buf, sendbuf, count * type_size);

  if (rank == root) {
    for (int i = 0; i < size; i++) {
      if (i != root) {
        void *recv_buf = malloc(count * type_size);
        if (!recv_buf) {
          free(temp_buf);
          throw std::runtime_error("Memory allocation failed");
        }

        MPI_Recv(recv_buf, count, datatype, i, 0, comm, MPI_STATUS_IGNORE);
        PerformOperation(recv_buf, temp_buf, count, datatype, op);
        free(recv_buf);
      }
    }
  } else {
    MPI_Send(sendbuf, count, datatype, root, 0, comm);
  }
  if (rank == root) {
    for (int i = 0; i < size; i++) {
      if (i != root) {
        MPI_Send(temp_buf, count, datatype, i, 1, comm);
      }
    }
    std::memcpy(recvbuf, temp_buf, count * type_size);
  } else {
    MPI_Recv(recvbuf, count, datatype, root, 1, comm, MPI_STATUS_IGNORE);
  }

  free(temp_buf);
}

template <typename T>
std::vector<T> BaranovACustomAllreduceMPI::GetVectorFromVariant(const InTypeVariant &variant) {
  try {
    return std::get<std::vector<T>>(variant);
  } catch (const std::bad_variant_access &) {
    throw std::runtime_error("Wrong variant type accessed");
  }
}

template std::vector<double> BaranovACustomAllreduceMPI::GetVectorFromVariant<double>(const InTypeVariant &variant);

BaranovACustomAllreduceMPI::BaranovACustomAllreduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  if (std::holds_alternative<std::vector<int>>(in)) {
    auto vec = std::get<std::vector<int>>(in);
    GetOutput() = InTypeVariant{std::vector<int>(vec.size(), 0)};
  } else if (std::holds_alternative<std::vector<float>>(in)) {
    auto vec = std::get<std::vector<float>>(in);
    GetOutput() = InTypeVariant{std::vector<float>(vec.size(), 0.0f)};
  } else {
    auto vec = std::get<std::vector<double>>(in);
    GetOutput() = InTypeVariant{std::vector<double>(vec.size(), 0.0)};
  }
}

bool BaranovACustomAllreduceMPI::ValidationImpl() {
  try {
    auto output = GetOutput();
    std::visit([](const auto &vec) {
      for (const auto &val : vec) {
        if (std::isnan(val) || std::isinf(val)) {
          throw std::runtime_error("Invalid value in output");
        }
      }
    }, output);
    return true;
  } catch (...) {
    return false;
  }
}

bool BaranovACustomAllreduceMPI::PreProcessingImpl() {
  return true;
}

bool BaranovACustomAllreduceMPI::RunImpl() {
  try {
    auto input = GetInput();
    auto output = GetOutput();

    if (std::holds_alternative<std::vector<int>>(input)) {
      auto data = std::get<std::vector<int>>(input);
      if (data.empty()) {
        GetOutput() = InTypeVariant{std::vector<int>{}};
        return true;
      }
      auto result_data = std::get<std::vector<int>>(output);
      CustomAllreduce(data.data(), result_data.data(), static_cast<int>(data.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                      0);

      GetOutput() = InTypeVariant{result_data};
    } else if (std::holds_alternative<std::vector<float>>(input)) {
      auto data = std::get<std::vector<float>>(input);
      if (data.empty()) {
        GetOutput() = InTypeVariant{std::vector<float>{}};
        return true;
      }
      auto result_data = std::get<std::vector<float>>(output);
      CustomAllreduce(data.data(), result_data.data(), static_cast<int>(data.size()), MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD, 0);
      GetOutput() = InTypeVariant{result_data};
    } else if (std::holds_alternative<std::vector<double>>(input)) {
      auto data = std::get<std::vector<double>>(input);
      if (data.empty()) {
        GetOutput() = InTypeVariant{std::vector<double>{}};
        return true;
      }
      auto result_data = std::get<std::vector<double>>(output);
      CustomAllreduce(data.data(), result_data.data(), static_cast<int>(data.size()), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD, 0);
      GetOutput() = InTypeVariant{result_data};
    }
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool BaranovACustomAllreduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_custom_allreduce
