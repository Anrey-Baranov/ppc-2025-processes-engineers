# Передача от всех одному и рассылка (allreduce)

- Студент: Синев Артём Александрович, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 3

## 1. Введение

Операция Allreduce является одной из фундаментальных коллективных операций в параллельных вычислениях. Она выполняет редукцию данных со всех процессов с применением указанной операции (например, суммирование) и рассылает результат всем процессам. Данная работа посвящена разработке пользовательской реализации операции Allreduce с использованием только базовых функций передачи сообщений MPI_Send и MPI_Recv. Особенностью реализации является использование алгоритма двоичного дерева для эффективной коммуникации между процессами.

## 2. Постановка задачи

Реализовать функцию MPI_Allreduce_custom, которая выполняет ту же задачу, что и стандартная MPI_Allreduce, но использует только MPI_Send и MPI_Recv.

**Входные данные:**
- sendbuf - указатель на буфер отправки данных (у каждого процесса свои данные)
- count - количество элементов в буфере
- datatype - тип данных (MPI_INT, MPI_FLOAT, MPI_DOUBLE)
- op - операция редукции (поддерживается только MPI_SUM)
- comm - коммуникатор MPI

**Выходные данные:**
- recvbuf - указатель на буфер приема результатов (у всех процессов одинаковый результат)

**Требования:**
- Использовать только MPI_Send и MPI_Recv
- Применить древовидную схему коммуникации
- Поддержка типов данных: MPI_INT, MPI_FLOAT, MPI_DOUBLE
- Поддержка операции: MPI_SUM

**Пример для 4 процессов:**

Процесс 0: [1, 2, 3]
Процесс 1: [4, 5, 6]
Процесс 2: [7, 8, 9]
Процесс 3: [10, 11, 12]

Результат у всех: [22, 26, 30] (сумма соответствующих элементов)

## 3. Описание базового алгоритма (SEQ версия)

Последовательная версия (SEQ) операции Allreduce является тривиальной, так как нет других процессов для коммуникации:

1. **Инициализация**: Копирование входных данных в выходной буфе 
2. **Завершение**: Возврат результата

Алгоритм состоит из следующих шагов:

```cpp 
bool SinevAAllreduceSEQ::RunImpl() {
  try {
    GetOutput() = GetInput();  // Простое копирование данных
    return true;
  } catch (const std::exception &) {
    return false;
  }
}
```

**Сложность алгоритма**

**Время:** 
- O(1) - простое копирование указателей

**Память:**
- O(N) - хранение входного вектора


## 4. Схема распараллеливания

### 4.1. Алгоритм двоичного дерева

Параллельная реализация использует алгоритм двоичного дерева (binary tree algorithm), который состоит из двух фаз:

- Фаза 1: Редукция (снизу вверх)
- Фаза 2: Рассылка (сверху вниз)

## 4.2. Ключевые особенности алгоритма

**Использование битовых операций**

```cpp 
int partner = rank ^ mask;       // Вычисление партнера с помощью XOR
if ((rank & mask) == 0) {       // Определение роли: получатель (0) или отправитель
```

**Двухфазный подход**

- Фаза редукции: Сбор данных к корню (процессу 0) с поэтапным суммированием
- Фаза рассылки: Распространение результата от корня всем процессам

**Распределение нагрузки**

- Каждый процесс участвует в вычислениях
- Работа распределяется равномерно между всеми процессами
- Коммуникации происходят параллельно на каждом этапе

**Роли процессов**
- **Процесс 0:** Координирует рассылку финального результата
- **Все процессы:** Вычисляют локальные суммы и участвуют в обмене данными
- **Каждая пара процессов:** Работает параллельно на каждом этапе

## 5. Детали реализации

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциоанльные тесты
- `tests/performance/main.cpp` - тесты производительности

**Ключевые классы:**
- `SinevAMinInVectorSEQ` - последовательная реализация
- `SinevAMinInVectorMPI` - параллельная MPI реализация

**Основные методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - подготовительные вычисления  
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - завершающая обработка

## Оптимизация структуры кода
Для улучшения читаемости и поддержки кода была проведена рефакторизация основной функции MpiAllreduceCustom. Были выделены три самостоятельные функции:

- `PerformReducePhase()` - отвечает за фазу редукции (сбор данных к корню)

- `BroadcastViaBinaryTree()` - выполняет рассылку результата через двоичное дерево

- `BroadcastRemainingProcesses()` - обеспечивает рассылку оставшимся процессам

**Логика разделения**:

- Фаза редукции выделена как самостоятельный процесс, так как она имеет четко определенную цель и алгоритм

- Рассылка разделена на две части для корректной работы с произвольным количеством процессов


## Особенности реализации

**Основная функция MPI_Allreduce_custom:**

```cpp
int SinevAAllreduce::MpiAllreduceCustom(const void *sendbuf, void *recvbuf, 
                                         int count, MPI_Datatype datatype,
                                         MPI_Op op, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  // Специальный случай: один процесс
  if (size == 1) {
    int type_size = GetTypeSize(datatype);
    size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    std::memcpy(recvbuf, sendbuf, total_size);
    return 0;
  }
  
  int type_size = GetTypeSize(datatype);
  int total_bytes = count * type_size;
  
  // Локальный буфер для данных
  std::vector<char> local_buffer(total_bytes);
  
  // Копируем входные данные
  if (sendbuf == MPI_IN_PLACE) {
    std::memcpy(local_buffer.data(), recvbuf, total_bytes);
  } else {
    std::memcpy(local_buffer.data(), sendbuf, total_bytes);
  }
  
  // === ФАЗА 1: РЕДУКЦИЯ (двоичное дерево) ===
  PerformReducePhase(rank, size, total_bytes, count, datatype, op, comm, local_buffer);
  
  // === ФАЗА 2: РАССЫЛКА (двоичное дерево) ===
  if (rank == 0) {
    std::memcpy(recvbuf, local_buffer.data(), total_bytes);
  }
  
  if (rank != 0 && sendbuf != MPI_IN_PLACE) {
    std::memcpy(recvbuf, sendbuf, total_bytes);
  }
  
  // Рассылка через двоичное дерево
  BroadcastViaBinaryTree(rank, size, count, datatype, comm, recvbuf);
  
  // Рассылка оставшимся процессам
  BroadcastRemainingProcesses(rank, size, count, datatype, comm, recvbuf);
  
  return 0;
}
```

**Функция редукции:**
```cpp
void PerformReducePhase(int rank, int size, int total_bytes, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                       std::vector<char>& local_buffer) {
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;

    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> recv_buffer(total_bytes);
        MPI_Recv(recv_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
        SinevAAllreduce::PerformOperation(local_buffer.data(), recv_buffer.data(), count, datatype, op);
      } else {
        MPI_Send(local_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
}
```

**Функция рассылки через двоичное дерево:**

```cpp
void BroadcastViaBinaryTree(int rank, int size, int count, 
                           MPI_Datatype datatype, MPI_Comm comm,
                           void* recvbuf) {
  int tree_size = 1;
  while (tree_size < size) {
    tree_size <<= 1;
  }
  
  for (int level = tree_size / 2; level > 0; level >>= 1) {
    if (rank < level) {
      int dest = rank + level;
      if (dest < size) {
        MPI_Send(recvbuf, count, datatype, dest, 1, comm);
      }
    } else if (rank < 2 * level && rank >= level) {
      int source = rank - level;
      if (source < size) {
        MPI_Recv(recvbuf, count, datatype, source, 1, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}
```

**Функция рассылки оставшимся процессам:**
```cpp
void BroadcastRemainingProcesses(int rank, int size, int count,
                                MPI_Datatype datatype, MPI_Comm comm,
                                void* recvbuf) {
  if (size <= 1) return;
  
  for (int step = 1; step < size; step *= 2) {
    if (rank < step) {
      int dest = rank + step;
      if (dest < size && dest >= step) {
        MPI_Send(recvbuf, count, datatype, dest, 2, comm);
      }
    } else if (rank < 2 * step && rank >= step) {
      int source = rank - step;
      if (source >= 0) {
        MPI_Recv(recvbuf, count, datatype, source, 2, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}
```

**Обработка разных типов данных:**

```cpp
void SinevAAllreduce::performOperation(void *inout, const void *in, 
                                      int count, MPI_Datatype datatype, 
                                      MPI_Op op) {
  if (op != MPI_SUM) return;
  
  if (datatype == MPI_INT) {
    performSumTemplate(static_cast<int*>(inout), 
                      static_cast<const int*>(in), count);
  } else if (datatype == MPI_FLOAT) {
    performSumTemplate(static_cast<float*>(inout), 
                      static_cast<const float*>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    performSumTemplate(static_cast<double*>(inout), 
                      static_cast<const double*>(in), count);
  }
}
```

**Шаблонная функция суммирования:**

```cpp
template <typename T>
void performSumTemplate(T *out, const T *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}
```

## Обработка граничных случаев

- **Один процесс**: Простое копирование данных без коммуникации
- **MPI_IN_PLACE**: Поддержка работы с одним буфером для ввода/вывода
- **Пустой вектор**: Корректная обработка нулевого размера данных
- **Произвольное количество процессов:** Рассылка корректно работает для любого числа процессов

## 6. Экспериментальное окружение

### 6.1 Аппаратное обеспечение/ОС:

- **Процессор:** Intel Core i7-13700HX
- **Ядра:** 16 физических ядер  
- **ОЗУ:** 8 ГБ 
- **ОС:** Kubuntu 24.04

### 6.2 Программный инструментарий

- **Компилятор:** g++ 13.3.0
- **Тип сборки:** Release
- **Стандарт C++:** C++20
- **MPI:** OpenMPI 4.1.6

### 6.3 Тестовое окружение

```bash
PPC_NUM_PROC=1,2,4,7,8
```

## 7. Результаты

### 7.1. Корректность работы

Все функциональные тесты пройдены успешно:
- 9 тестовых конфигураций: Разные размеры (1, 10, 100, 1000 элементов) и типы данных (int, float, double)
- SEQ vs MPI: Обе реализации выдают идентичные результаты
- MPI тесты: Правильная работа для 1, 2, 4 процессов


### 7.2. Производительность

**Время выполнения (секунды) для вектора из 10,000,000 элементов типа double:**

| Версия | Количество процессов | Task Run время |
|--------|---------------------|----------------|
| SEQ    | 1                   | 	0.0041      |
| MPI    | 1                   | 	0.0041      | 
| MPI    | 2                   | 	0.0782      |
| MPI    | 4                   | 0.1381       |
| MPI    | 7                   | 	0.2008      | 
| MPI    | 8                   | 	0.2208      |

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 1.00×     | 100%           |
| 2                   | 0.052×     | 2.6%           |
| 4                   | 0.030×     | 0.75%           |
| 7                   | 0.020×     | 0.29%           |
| 8                   | 0.019×     | 0.24%          |


**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ эффективности

- **Масштабируемость**: Производительность продолжает линейно ухудшаться с увеличением числа процессов

**Ожидаемое поведение vs Фактическое:**


    Ожидалось: Ускорение с увеличением числа процессов

    Фактически: Замедление с увеличением числа процессов

**Причины замедления:**

- **Алгоритмическая сложность:** Двоичное дерево требует O(log₂P) шагов, но на каждом шаге передаются ВСЕ данные

- **Объем передаваемых данных:** Для 10M double - 80 МБ на каждом шаге коммуникации

- **Коммуникационные затраты:** Доминируют над вычислительными

- **Отсутствие перекрытия:** Вычисления и коммуникации не перекрываются

- **Использование блокирующих операций:** Каждая операция Send/Recv блокирует выполнение


## 8. Выводы

### 8.1. Достигнутые результаты

**Корректность реализации:**

- Успешное прохождение всех функциональных тестов
- Поддержка трех типов данных: int, float, double
- Правильная работа алгоритма двоичного дерева
- Обработка граничных случаев (1 процесс, пустые данные)

**Соответствие требованиям:**

- Использование только MPI_Send и MPI_Recv
- Реализация древовидной схемы коммуникации
- Поддержка операции MPI_SUM

**Архитектурные решения:**

- Модульная структура с разделением SEQ и MPI версий
- Использование std::variant для типобезопасности
- Шаблонные функции для поддержки разных типов данных

### 8.2. Ограничения и проблемы

**Производительность:**

- Алгоритм не масштабируется с увеличением числа процессов
- начительные коммуникационные затраты
- Отсутствие ускорения при распараллеливании
- Большой объем данных передается на каждом шаге

**Алгоритмические ограничения:**

- На каждом шаге передаются все данные
- Нет перекрытия вычислений и коммуникаций
- Используются только блокирующие операции
- Для произвольного количества процессов требуется дополнительная логика рассылки

**Функциональные ограничения**:

- Поддерживается только операция MPI_SUM
- Нет поддержки пользовательских операций
- Ограниченный набор типов данных

### 8.3. Особенности реализации

**Положительные аспекты:**

- Алгоритм двоичного дерева: Логарифмическая сложность по количеству шагов
- Распределенная работа: Все процессы участвуют в вычислениях

**Проблемные аспекты:**

- Коммуникационные затраты: Доминируют над вычислительными
- Объем данных: Передача всех данных на каждом шаге

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Документация MPI: https://www.open-mpi.org/
3. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
#include "sinev_a_allreduce/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstring>
#include <variant>
#include <vector>

#include "sinev_a_allreduce/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_allreduce {

SinevAAllreduce::SinevAAllreduce(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool SinevAAllreduce::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized == 1;
}

bool SinevAAllreduce::PreProcessingImpl() {
  return true;
}

int SinevAAllreduce::GetTypeSize(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 1;
}

namespace {
template <typename T>
void PerformSumTemplate(T *out, const T *in, int count) {
  for (int i = 0; i < count; i++) {
    out[i] += in[i];
  }
}
}  // namespace

void SinevAAllreduce::PerformOperation(void *inout, const void *in, int count, MPI_Datatype datatype, MPI_Op op) {
  if (op != MPI_SUM) {
    return;
  }

  if (datatype == MPI_INT) {
    PerformSumTemplate(static_cast<int *>(inout), static_cast<const int *>(in), count);
  } else if (datatype == MPI_FLOAT) {
    PerformSumTemplate(static_cast<float *>(inout), static_cast<const float *>(in), count);
  } else if (datatype == MPI_DOUBLE) {
    PerformSumTemplate(static_cast<double *>(inout), static_cast<const double *>(in), count);
  }
}

namespace {

void PerformReducePhase(int rank, int size, int total_bytes, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                       std::vector<char>& local_buffer) {
  int mask = 1;
  while (mask < size) {
    int partner = rank ^ mask;

    if (partner < size) {
      if ((rank & mask) == 0) {
        std::vector<char> recv_buffer(total_bytes);
        MPI_Recv(recv_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm, MPI_STATUS_IGNORE);
        SinevAAllreduce::PerformOperation(local_buffer.data(), recv_buffer.data(), count, datatype, op);
      } else {
        MPI_Send(local_buffer.data(), total_bytes, MPI_BYTE, partner, 0, comm);
        break;
      }
    }
    mask <<= 1;
  }
}

void BroadcastViaBinaryTree(int rank, int size, int count, 
                           MPI_Datatype datatype, MPI_Comm comm,
                           void* recvbuf) {
  int tree_size = 1;
  while (tree_size < size) {
    tree_size <<= 1;
  }
  
  for (int level = tree_size / 2; level > 0; level >>= 1) {
    if (rank < level) {
      int dest = rank + level;
      if (dest < size) {
        MPI_Send(recvbuf, count, datatype, dest, 1, comm);
      }
    } else if (rank < 2 * level && rank >= level) {
      int source = rank - level;
      if (source < size) {
        MPI_Recv(recvbuf, count, datatype, source, 1, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}

void BroadcastRemainingProcesses(int rank, int size, int count,
                                MPI_Datatype datatype, MPI_Comm comm,
                                void* recvbuf) {
  if (size <= 1) return;
  
  for (int step = 1; step < size; step *= 2) {
    if (rank < step) {
      int dest = rank + step;
      if (dest < size && dest >= step) {
        MPI_Send(recvbuf, count, datatype, dest, 2, comm);
      }
    } else if (rank < 2 * step && rank >= step) {
      int source = rank - step;
      if (source >= 0) {
        MPI_Recv(recvbuf, count, datatype, source, 2, comm, MPI_STATUS_IGNORE);
      }
    }
  }
}

}  // namespace

int SinevAAllreduce::MpiAllreduceCustom(const void *sendbuf, void *recvbuf, int count, 
                                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size == 1) {
    int type_size = GetTypeSize(datatype);
    size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    std::memcpy(recvbuf, sendbuf, total_size);
    return 0;
  }

  int type_size = GetTypeSize(datatype);
  int total_bytes = count * type_size;

  std::vector<char> local_buffer(total_bytes);

  if (sendbuf == MPI_IN_PLACE) {
    std::memcpy(local_buffer.data(), recvbuf, total_bytes);
  } else {
    std::memcpy(local_buffer.data(), sendbuf, total_bytes);
  }

  PerformReducePhase(rank, size, total_bytes, count, datatype, op, comm, local_buffer);

  if (rank == 0) {
    std::memcpy(recvbuf, local_buffer.data(), total_bytes);
  }

  if (rank != 0 && sendbuf != MPI_IN_PLACE) {
    std::memcpy(recvbuf, sendbuf, total_bytes);
  }

  BroadcastViaBinaryTree(rank, size, count, datatype, comm, recvbuf);

  BroadcastRemainingProcesses(rank, size, count, datatype, comm, recvbuf);

  return 0;
}

bool SinevAAllreduce::RunImpl() {
  auto &input_variant = GetInput();
  auto &output_variant = GetOutput();

  try {
    if (std::holds_alternative<std::vector<int>>(input_variant)) {
      auto &input = std::get<std::vector<int>>(input_variant);
      auto &output = std::get<std::vector<int>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<float>>(input_variant)) {
      auto &input = std::get<std::vector<float>>(input_variant);
      auto &output = std::get<std::vector<float>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_FLOAT, MPI_SUM,
                         MPI_COMM_WORLD);

    } else if (std::holds_alternative<std::vector<double>>(input_variant)) {
      auto &input = std::get<std::vector<double>>(input_variant);
      auto &output = std::get<std::vector<double>>(output_variant);

      if (output.size() != input.size()) {
        output.resize(input.size());
      }

      MpiAllreduceCustom(input.data(), output.data(), static_cast<int>(input.size()), MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD);
    }

    return true;
  } catch (...) {
    return false;
  }
}

bool SinevAAllreduce::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_allreduce

```