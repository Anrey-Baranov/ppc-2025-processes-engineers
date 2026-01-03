# "Топология сетей передачи данных "Линейка"", вариант № 6
### Студент: Редькина Алина Александровна
### Группа: 3823Б1ПР1
### Преподаватель: Сысоев Александр Владимирович, доцент


## Введение

  Топология сети передачи данных определяет способ соединения процессов между собой и порядок обмена сообщениями. Одной из наиболее простых и наглядных топологий является линейка (цепочка), в которой каждый процесс взаимодействует только со своими ближайшими соседями.
  В рамках параллельного программирования с использованием MPI данная топология представляет практический интерес, поскольку позволяет реализовать маршрутизацию сообщений без применения встроенных средств описания топологий, таких как `MPI_Cart_Create` и `MPI_Graph_Create`. Это требует явного управления передачей данных между процессами и более глубокого понимания модели передачи сообщений MPI.

---

## Постановка задачи

**Цель работы:**  
  Реализовать виртуальную топологию типа «линейка» с использованием MPI и обеспечить возможность передачи данных от произвольного процесса-источника к произвольному процессу-получателю.

**Определение задачи:**  
  Необходимо передать вектор `data` от процесса `start` к процессу `end`, используя линейную топологию сети, в которой каждый процесс может обмениваться данными только с соседними процессами.

**Ограничения:**
  - Используется модель передачи сообщений MPI.
  - Запрещено использование `MPI_Cart_Create` и `MPI_Graph_Create`.
  - Передача должна происходить пошагово через соседние процессы.
  - Передача должна быть корректной для любого допустимого `stast` и `end`.
  - Поддерживается случай `start == end`.

---

## Описание алгоритма (последовательная версия)
  В последовательной версии задача упрощается, так как отсутствует распределение процессов.

Алгоритм:
  1. Получить входные данные.
  2. Скопировать вектор `data` в выходной вектор без изменений.
  3. Вернуть результат.

  Последовательная версия используется для проверки корректности и сравнения с MPI-реализацией.

### Код последовательной реализации

```cpp
bool RedkinaARulerSEQ::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = input.data;
  return true;
}
```

---

## Схема распараллеливания (MPI)

В параллельной версии процессы образуют логическую цепочку: 0 — 1 — 2 — ... — (P-1). 
Каждый процесс знает только:
  - свой номер (rank),
  - левого соседа (rank - 1),
  - правого соседа (rank + 1).

### Основные этапы алгоритма

1. Инициализация:
  - Все процессы работают в коммуникаторе `MPI_COMM_WORLD`.
  - Определяются номера процессов-источника и назначения.

2. Определение направления передачи:
  - Если `end > start`, данные передаются вправо.
  - Если `end < start`, данные передаются влево.

3. Обработка случая `start == end`:
  - Данные не передаются по цепочке.
  - Процесс-источник сразу формирует результат и рассылает его всем процессам с помощью `MPI_Bcast`.

4. Маршрутизация данных:
  - Процесс-источник отправляет данные соседу в выбранном направлении.
  - Каждый промежуточный процесс: принимает данные от одного соседа, передаёт их следующему.
  - Процесс-получатель сохраняет данные как результат.

5. Синхронизация результата:
  - После достижения процесса dest результат рассылается всем процессам с помощью MPI_Bcast.

Передача вектора осуществляется в два этапа:
  - сначала передаётся размер данных,
  - затем сами данные.


### Код параллельной реализации

```cpp
bool RedkinaARulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto& input = GetInput();
  const int start = input.start;
  const int end = input.end;

  if (start < 0 || start >= size || end < 0 || end >= size) {
    return false;
  }

  if (start == end) {
    HandleSameStartEnd(rank, start, input.data, GetOutput());
    return true;
  }

  const bool go_right = end > start;
  const int left_n = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right_n = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

  RouteDataStartEnd(rank, start, end, go_right, left_n, right_n, input.data, GetOutput());

  BroadcastEndResult(rank, end, GetOutput());
  return true;
}
```

---

## Экспериментальные результаты

### Окружение
| Параметр | Значение |
|-----------|-----------|
| Процессор | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics |
| Операционная система | Windows 11 |
| Компилятор | g++ 13.3.0 |
| Тип сборки | Release |
| Число процессов | 2 |

### Проверка корректности
Были проведены функциональные, граничные и расширенные тесты:
  - Передача между соседними процессами (0 → 1, 1 → 0)
  - Передача с несколькими промежуточными процессами
  - Случай `start == end`
  - Передача одного элемента
  - Передача больших векторов
  - Передача отрицательных, нулевых и граничных значений (`INT_MIN`, `INT_MAX`)
  - Передача векторов с одинаковыми элементами

**Результат:** Все тесты успешно пройдены, последовательная и MPI-версии возвращают одинаковые значения.

### Оценка производительности
  Для оценки производительности использовались тесты с вектором из 100000000 элементов.

**Время выполнения task_run**

| Режим | Процессы | Время, с |
|-------|----------|----------|
| mpi   | 2        | 0.025    |
| mpi   | 3        | 0.041    |
| mpi   | 4        | 0.060    |

**Время выполнения pipeline**

| Режим | Процессы | Время, с |
|-------|----------|----------|
| mpi   | 2        | 0.046    |
| mpi   | 3        | 0.065    |
| mpi   | 4        | 0.083    |

**Результат:** Все тесты успешно пройдены.

**Наблюдения:**
  - Время выполнения MPI-версии увеличивается с ростом числа процессов между `start` и `end`.
  - Основные накладные расходы связаны с пересылкой больших объёмов данных.
  - Алгоритм не предназначен для ускорения вычислений, а демонстрирует корректную маршрутизацию сообщений.

---

## Выводы

  1. **Корректность:** реализованная MPI-версия корректно передаёт данные между любыми процессами в линейной топологии.
  2. **Соответствие требованиям:** в реализации не используются `MPI_Cart_Create` и `MPI_Graph_Create`. Топология формируется вручную.  
  3. **Гибкость:** алгоритм поддерживает передачу данных в обоих направлениях и корректно обрабатывает случай `start == end`. 

---

## Источники

  1. Лекции Сысоева Александра Владимировича.

## Приложения

**common.hpp**
```cpp
#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_ruler {

struct RulerMessage {
  int start{};
  int end{};
  std::vector<int> data;
};

using InType = RulerMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<int, RulerMessage>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_ruler
```

**ops_seq.hpp**
```cpp
#pragma once

#include "redkina_a_ruler/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_ruler {

class RedkinaARulerSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaARulerSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_ruler
```

**ops_mpi.hpp**
```cpp
#pragma once

#include "redkina_a_ruler/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_ruler {

class RedkinaARulerMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RedkinaARulerMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_ruler
```

**ops_seq.cpp**
```cpp
#include "redkina_a_ruler/seq/include/ops_seq.hpp"

#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"

namespace redkina_a_ruler {

RedkinaARulerSEQ::RedkinaARulerSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RedkinaARulerSEQ::ValidationImpl() {
  const auto& input = GetInput();
  return input.start >= 0 && input.end >= 0 && !input.data.empty();
}

bool RedkinaARulerSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool RedkinaARulerSEQ::RunImpl() {
  const auto& input = GetInput();
  GetOutput() = input.data;
  return true;
}

bool RedkinaARulerSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_ruler
```

**ops_mpi.cpp**
```cpp
#include "redkina_a_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "redkina_a_ruler/common/include/common.hpp"

namespace redkina_a_ruler {

namespace {

bool IsInStartEndChain(const int rank, const int start, const int end, const bool go_right) {
  if (go_right) {
    return (rank > start && rank <= end);
  }
  return (rank < start && rank >= end);
}

void SendSizeAndData(const int end_rank, const uint64_t data_size, const std::vector<int>& data,
                     const int size_tag = 0, const int data_tag = 1) {
  if (end_rank == MPI_PROC_NULL) {
    return;
  }

  MPI_Send(&data_size, 1, MPI_UINT64_T, end_rank, size_tag, MPI_COMM_WORLD);
  if (data_size > 0U) {
    MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, end_rank, data_tag, MPI_COMM_WORLD);
  }
}

void ReceiveSizeAndData(const int start_rank, uint64_t& data_size, std::vector<int>& data,
                        const int size_tag = 0, const int data_tag = 1) {
  if (start_rank == MPI_PROC_NULL) {
    return;
  }

  MPI_Recv(&data_size, 1, MPI_UINT64_T, start_rank, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (data_size > 0U) {
    data.resize(static_cast<std::size_t>(data_size));
    MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT,
             start_rank, data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void BroadcastEndResult(const int rank, const int root, std::vector<int>& output) {
  uint64_t data_size = 0;

  if (rank == root) {
    data_size = static_cast<uint64_t>(output.size());
  }

  MPI_Bcast(&data_size, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  if (rank != root) {
    output.resize(static_cast<std::size_t>(data_size));
  }

  if (data_size > 0U) {
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, root, MPI_COMM_WORLD);
  }
}

void HandleSameStartEnd(const int rank, const int start, const std::vector<int>& input_data,
                        std::vector<int>& output) {
  if (rank == start) {
    output = input_data;
  }

  auto data_size = static_cast<uint64_t>(input_data.size());
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, start, MPI_COMM_WORLD);

  if (rank != start) {
    output.resize(static_cast<std::size_t>(data_size));
  }

  if (data_size > 0U) {
    MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, start, MPI_COMM_WORLD);
  }
}

void RouteDataStartEnd(const int rank, const int start, const int end, const bool go_right,
                       const int left_n, const int right_n,
                       const std::vector<int>& input_data, std::vector<int>& output) {
  std::vector<int> buffer;
  uint64_t data_size = 0;

  if (rank == start) {
    buffer = input_data;
    data_size = static_cast<uint64_t>(buffer.size());
    const int target = go_right ? right_n : left_n;
    SendSizeAndData(target, data_size, buffer);
  }

  if (!IsInStartEndChain(rank, start, end, go_right)) {
    return;
  }

  const int recv_from = go_right ? left_n : right_n;
  const int send_to = go_right ? right_n : left_n;

  ReceiveSizeAndData(recv_from, data_size, buffer);

  if (rank == end) {
    output = buffer;
  } else {
    SendSizeAndData(send_to, data_size, buffer);
  }
}

}  // namespace

RedkinaARulerMPI::RedkinaARulerMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaARulerMPI::ValidationImpl() {
  const auto& input = GetInput();
  return input.start >= 0 && input.end >= 0 && !input.data.empty();
}

bool RedkinaARulerMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RedkinaARulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto& input = GetInput();
  const int start = input.start;
  const int end = input.end;

  if (start < 0 || start >= size || end < 0 || end >= size) {
    return false;
  }

  if (start == end) {
    HandleSameStartEnd(rank, start, input.data, GetOutput());
    return true;
  }

  const bool go_right = end > start;
  const int left_n = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right_n = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

  RouteDataStartEnd(rank, start, end, go_right, left_n, right_n, input.data, GetOutput());

  BroadcastEndResult(rank, end, GetOutput());
  return true;
}

bool RedkinaARulerMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_ruler
```