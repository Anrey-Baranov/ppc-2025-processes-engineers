# Отчет по лабораторной работе №2
## Работу выполнил студент группы 3823Б1ПР1, Журин Иван Эдуардович
## Вариант № 7. Кольцо.

**Преподаватель:** Сысоев Александр Владимирович, лектор, доцент кафедры высокопроизводительных вычислений и системного программирования

---

## Введение

**Цель работы:** исследование методов параллельного программирования с использованием технологии MPI (Message Passing Interface) на примере задачи сетевая топология кольцо.

**Актуальность:** с ростом объемов обрабатываемых данных последовательные алгоритмы становятся недостаточно эффективными. Применение параллельных технологий позволяет обрабатывать большие массивы данных без значительного увеличения времени вычислений. Кольцевая топология представляет собой фундаментальную модель коммуникации, которая используется во многих параллельных алгоритмах.

В рамках работы требовалось реализовать два варианта алгоритма:
- **Последовательный (SEQ)** — обработка данных в одном процессе
- **Параллельный (MPI)** — распределение работы между несколькими процессами с последующим объединением результатов

---

## Постановка задачи

Разработать программу, реализующую передачу массива целых чисел от процесса-источника к процессу-получателю в кольцевой топологии.  Требуется реализовать:

- **Последовательный алгоритм (SEQ)** — вычисления в одном процессе.
- **Параллельный алгоритм (MPI)** — распределение элементов вектора между процессами и объединение частичных сумм.

---

## Описание алгоритма

### Последовательный алгоритм (SEQ)

1. Проверка корректности входных данных (source ≥ 0, dest ≥ 0)
2. Прямое копирование входного массива в выходной буфер
3. Имитация задержки для корректного сравнения с параллельной версией

### Параллельный алгоритм (MPI)

1. **Инициализация MPI:** получение ранга процесса и общего количества процессов
2. **Валидация:** проверка корректности входных параметров
3. **Обработка особых случаев:** 
    - Если количество процессов = 1, то данные копируются в выходной буфер
    - Если источник и получатель совпадают, данные рассылаются всем процессам
4. **Корректировка параметров:** приведение номера источника и получателя к диапазону [0, world_size-1]
5. **Определение направления передачи:** вычисление количества шагов `steps` для движения в указанном направлении (`go_clockwise`)
6. **Передача данных:** Последовательная передача по кольцу через промежуточные процессы:
   - Вычисление количества шагов `steps` на основе направления
   - Циклическая передача данных от процесса к процессу с использованием `MPI_Send`/`MPI_Recv`
   - Каждый процесс выполняет роль либо отправителя, либо получателя, либо просто участвует в барьере синхронизации
7. **Рассылка результата:** После получения данных процессом-получателем результат рассылается всем процессам через `MPI_Bcast`

---

## Описание схемы параллельного алгоритма

Схема передачи данных для произвольного количества процессов:
1. **Инициализация:** Каждый процесс получает свой `rank` и `world_size`
2. **Корректировка адресов:** Приведение `source` и `dest` к диапазону `[0, world_size-1]` с помощью операции `% world_size`
3. **Расчет параметров передачи:**
   - Определение направления: `direction = go_clockwise ? 1 : -1`
   - Расчет количества шагов: `steps = (direction == 1) ? (dest - source + world_size) % world_size : (source - dest + world_size) % world_size`
4. **Последовательная передача по кольцу:** Для каждого шага от 0 до `steps-1`:
   - Определение отправителя: `sender = (source + step * direction + world_size) % world_size`
   - Определение получателя: `receiver = (sender + direction + world_size) % world_size`
   - Отправитель передает данные получателю через `MPI_Send`
   - Получатель принимает данные через `MPI_Recv`
5. **Фиксация результата:** Процесс с `rank == dest` сохраняет полученные данные в выходной буфер
6. **Широковещательная рассылка:** Процесс-получатель рассылает результат всем процессам через `MPI_Bcast`

---

## Описание программной реализации

### Параллельная реализация (MPI)
**Архитектура коммуникации:**
- Равноправные процессы в коммуникаторе MPI_COMM_WORLD
- Последовательная передача данных по цепочке процессов
- Децентрализованные вычисления

### Фазы выполнения MPI-алгоритма
#### ValidationImpl()
- Локальная проверка корректности входных данных
- Подтверждение неотрицательности source и dest

#### PreProcessingImpl()
- Инициализация выходного буфера
- Подготовка к приему данных

#### RunImpl()
- Определение параметров передачи — расчет эффективных source и dest
- Обработка особых случаев — world_size = 1 или source == dest
- Определение направления передачи — использование параметра go_clockwise
- Передача данных — последовательная передача по кольцу
- Рассылка результата — MPI_Bcast от процесса-получателя

#### PostProcessingImpl()
- Каждый процесс имеет готовый результат
- Дополнительные коммуникации не требуются

### Ключевые особенности реализации

- **Поддержка двух направлений:** передача по часовой и против часовой стрелки
- **Единый алгоритм для любого количества процессов:** реализация использует универсальный подход с циклом по шагам передачи, что упрощает код 
- **Гибкая обработка параметров:** приведение source/dest к диапазону с помощью операции модуля
- **Согласованность данных:** идентичные результаты на всех процессах благодаря MPI_Bcast
- **Масштабируемость:** поддержка произвольного числа процессов

---

## Тестирование

### Результаты производительности (100.000.000 элементов)

#### Время выполнения `pipeline` (секунды)

| Процессы |   Время    | Ускорение | Эффективность |
|----------|------------|-----------|---------------|
| 1        | 0.04030774 | 1.000     | 1.000         |
| 2        | 0.28419660 | 0.142     | 0.071         |
| 3        | 0.48768196 | 0.083     | 0.028         |
| 4        | 0.71909900 | 0.056     | 0.014         |
| 5        | 0.87713252 | 0.046     | 0.009         |
| 6        | 0.97789604 | 0.041     | 0.007         |
| 7        | 1.18403688 | 0.034     | 0.005         |
| 8        | 1.32167618 | 0.030     | 0.004         |
| 16       | 4.07143968 | 0.010     | 0.001         |

#### Время выполнения `task_run` (секунды)

| Процессы |   Время    | Ускорение | Эффективность |
|----------|------------|-----------|---------------|
| 1        | 0.03869810 | 1.000     | 1.000         |
| 2        | 0.27113278 | 0.143     | 0.071         |
| 3        | 0.44362454 | 0.087     | 0.029         |
| 4        | 0.60251276 | 0.064     | 0.016         |
| 5        | 0.90933002 | 0.043     | 0.009         |
| 6        | 0.78620828 | 0.049     | 0.008         |
| 7        | 0.89677030 | 0.043     | 0.006         |
| 8        | 0.99209902 | 0.039     | 0.005         |
| 16       | 4.39398280 | 0.009     | 0.001         |

### Анализ результатов

1. **Производительность снижается с увеличением числа процессов** - наблюдается отрицательное ускорение (speedup < 1)
2. **Эффективность крайне низкая** - ниже 0.1 уже при 2 процессах
3. **Задача не подходит для распараллеливания** через MPI в текущей реализации

### Проверка корректности
Все функциональные тесты успешно пройдены, что подтверждает:
1. **Корректность передачи данных:** данные доставляются от источника к получателю без искажений при любом количестве процессов
2. **Обработка особых случаев:** корректная работа при world_size = 1 и source = dest
3. **Согласованность результатов:** все процессы получают одинаковый результат после широковещательной рассылки
4. **Масштабируемость алгоритма:** корректная работа с различным количеством процессов (1-16)
5. **Обработка граничных значений:** работа с пустыми массивами, экстремальными значениями (INT_MAX, INT_MIN)

---

## Заключение

В ходе лабораторной работы были успешно реализованы последовательный и параллельный алгоритмы передачи данных в кольцевой топологии. Параллельная реализация на основе MPI корректно работает для любого количества процессов и эффективно обрабатывает все граничные случаи.
В данной задаче при росте процессов растут накладные расходы, что приводит к замедлению. 

Алгоритм демонстрирует корректную работу и масштабируемость, однако ввиду коммуникационной природы задачи и малого объема вычислений, увеличение числа процессов приводит к росту времени выполнения.
---

## Приложение


# **common.hpp**

#ifndef ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_
#define ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_

#include <tuple>
#include <vector>

namespace zhurin_i_ring_topology {

struct RingMessage {
  int source = 0;
  int dest = 0;
  std::vector<int> data;
  bool go_clockwise = true;
};

using InType = RingMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<int, RingMessage>;

}  // namespace zhurin_i_ring_topology

#endif  // ZHURIN_I_RING_TOPOLOGY_COMMON_INCLUDE_COMMON_HPP_

---

# **ops_mpi.hpp**

#ifndef ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_
#define ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_

#include "task/include/task.hpp"
#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

using BaseTask = ppc::task::Task<InType, OutType>;

class ZhurinIRingTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZhurinIRingTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zhurin_i_ring_topology

#endif  // ZHURIN_I_RING_TOPOLOGY_MPI_INCLUDE_OPS_MPI_HPP_

---

# **ops_mpi.cpp**

#include "zhurin_i_ring_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

namespace {

void SendData(int rank, int sender, int receiver, uint64_t data_size, const std::vector<int> &data) {
  if (rank != sender) {
    return;
  }

  MPI_Send(&data_size, 1, MPI_UINT64_T, receiver, 0, MPI_COMM_WORLD);
  if (data_size > 0) {
    MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, receiver, 1, MPI_COMM_WORLD);
  }
}

void ReceiveData(int rank, int sender, int receiver, uint64_t &data_size, std::vector<int> &buffer) {
  if (rank != receiver) {
    return;
  }

  MPI_Recv(&data_size, 1, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  buffer.resize(static_cast<size_t>(data_size));
  if (data_size > 0) {
    MPI_Recv(buffer.data(), static_cast<int>(data_size), MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void BroadcastToAll(int root, std::vector<int> &output) {
  auto data_size = static_cast<uint64_t>(output.size());
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  if (data_size > 0) {
    output.resize(static_cast<size_t>(data_size));
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, root, MPI_COMM_WORLD);
  }
}

}  // namespace

ZhurinIRingTopologyMPI::ZhurinIRingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZhurinIRingTopologyMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool ZhurinIRingTopologyMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZhurinIRingTopologyMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();

  int source = input.source % world_size;
  int dest = input.dest % world_size;

  if (source == dest) {
    if (rank == source) {
      GetOutput() = input.data;
    }
    BroadcastToAll(source, GetOutput());
    return true;
  }

  int direction = input.go_clockwise ? 1 : -1;
  int steps = 0;

  if (direction == 1) {
    steps = (dest - source + world_size) % world_size;
  } else {
    steps = (source - dest + world_size) % world_size;
  }

  std::vector<int> buffer;
  uint64_t data_size = 0;

  for (int step = 0; step < steps; step++) {
    int sender = (source + step * direction + world_size) % world_size;
    int receiver = (sender + direction + world_size) % world_size;

    if (step == 0) {
      SendData(rank, sender, receiver, static_cast<uint64_t>(input.data.size()), input.data);
    } else {
      SendData(rank, sender, receiver, data_size, buffer);
    }

    ReceiveData(rank, sender, receiver, data_size, buffer);

    if (receiver == dest && rank == dest) {
      GetOutput() = buffer;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  BroadcastToAll(dest, GetOutput());

  return true;
}

bool ZhurinIRingTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zhurin_i_ring_topology

---

# **ops_seq.hpp**

#ifndef ZHURIN_I_RING_TOPOLOGY_SEQ_INCLUDE_OPS_SEQ_HPP_
#define ZHURIN_I_RING_TOPOLOGY_SEQ_INCLUDE_OPS_SEQ_HPP_

#include "task/include/task.hpp"
#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

using BaseTask = ppc::task::Task<InType, OutType>;

class ZhurinIRingTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZhurinIRingTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zhurin_i_ring_topology

#endif  // ZHURIN_I_RING_TOPOLOGY_SEQ_INCLUDE_OPS_SEQ_HPP_

---

# **ops_seq.cpp**

#include "zhurin_i_ring_topology/seq/include/ops_seq.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "zhurin_i_ring_topology/common/include/common.hpp"

namespace zhurin_i_ring_topology {

ZhurinIRingTopologySEQ::ZhurinIRingTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZhurinIRingTopologySEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.source >= 0 && input.dest >= 0;
}

bool ZhurinIRingTopologySEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZhurinIRingTopologySEQ::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = input.data;

  if (input.source != input.dest) {
    int distance = std::abs(input.dest - input.source);
    std::chrono::microseconds delay(static_cast<int64_t>(distance));
    std::this_thread::sleep_for(delay);
  }

  return true;
}

bool ZhurinIRingTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zhurin_i_ring_topology
