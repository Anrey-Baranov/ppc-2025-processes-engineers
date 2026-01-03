# Быстрая сортировка с простым слиянием

- Студент: Синев Артём Александрович, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 14

## 1. Введение

Задача сортировки является одной из фундаментальных в информатике и вычислительной технике. Данная работа посвящена разработке и реализации параллельного алгоритма быстрой сортировки с простым слиянием с использованием технологии MPI. Алгоритм сочетает преимущества быстрой сортировки (эффективность в среднем случае) с простым слиянием для объединения отсортированных подмассивов, обеспечивая масштабируемость на многопроцессорных системах.

## 2. Постановка задачи

Дан вектор целых чисел. Требуется отсортировать его в порядке возрастания.

**Входные данные:**
- vector - одномерный вектор целых чисел произвольной длины N

**Выходные данные:**
- sorted_vector - отсортированный вектор той же длины

**Ограничения:**
- Вектор должен содержать хотя бы один элемент
- Элементы вектора - целые числа (int)
- Допускаются отрицательные значения и дубликаты

**Пример:**

Входные данные: vector = [5, 3, 8, 1, 9, 2]
Выходные данные: sorted_vector = [1, 2, 3, 5, 8, 9]

## 3. Описание базового алгоритма (последовательная версия)

Последовательный алгоритм представляет собой модифицированную версию быстрой сортировки, где после разделения и рекурсивной сортировки подмассивов выполняется простое слияние для улучшения производительности на частично отсортированных данных.

Алгоритм состоит из следующих шагов:

1. **Разделение (Partition)**: 
    - Выбор опорного элемента (pivot) - среднего элемента подмассива
    - Перестановка элементов так, чтобы все элементы ≤ pivot оказались слева, а > pivot - справа
    - Возврат индекса опорного элемента
2. **Итеративная сортировка с использованием стека**:
   - Использование явного стека для хранения границ подмассивов
   - Обработка подмассивов в цикле до опустошения стека
   - Оптимизация порядка обработки (сначала меньший подмассив)
3. **Простое слияние (Simple Merge)**:
   - После сортировки обоих подмассивов выполняется их слияние в один отсортированный массив
   - Используется временный буфер для слияния

### Реализация ключевых функций

```cpp
// Статическая функция разделения
static int Partition(std::vector<int>& arr, int left, int right) {
  int pivot_index = left + (right - left) / 2;
  int pivot_value = arr[pivot_index];
  
  std::swap(arr[pivot_index], arr[left]);
  
  int i = left + 1;
  int j = right;
  
  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) i++;
    while (i <= j && arr[j] > pivot_value) j--;
    
    if (i < j) std::swap(arr[i], arr[j]);
  }
  
  std::swap(arr[left], arr[j]);
  return j;
}

// Статическая функция простого слияния
static void SimpleMerge(std::vector<int>& arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);
  
  int i = left, j = mid + 1, k = 0;
  
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) temp[k++] = arr[i++];
    else temp[k++] = arr[j++];
  }
  
  while (i <= mid) temp[k++] = arr[i++];
  while (j <= right) temp[k++] = arr[j++];
  
  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

// Итеративная версия быстрой сортировки с использованием стека
static void QuickSortWithSimpleMerge(std::vector<int>& arr, int left, int right) {
  struct Range { int left; int right; };
  std::stack<Range> stack;
  stack.push({left, right});
  
  while (!stack.empty()) {
    Range current = stack.top();
    stack.pop();
    
    if (current.left >= current.right) continue;
    
    int pivot_index = Partition(arr, current.left, current.right);
    
    // Оптимизация: сначала добавляем меньший подмассив
    int left_size = pivot_index - current.left;
    int right_size = current.right - pivot_index;
    
    if (left_size > 1 && right_size > 1) {
      if (left_size > right_size) {
        stack.push({pivot_index + 1, current.right});
        stack.push({current.left, pivot_index - 1});
      } else {
        stack.push({current.left, pivot_index - 1});
        stack.push({pivot_index + 1, current.right});
      }
    } else if (left_size > 1) {
      stack.push({current.left, pivot_index - 1});
    } else if (right_size > 1) {
      stack.push({pivot_index + 1, current.right});
    }
    
    SimpleMerge(arr, current.left, pivot_index, current.right);
  }
}
```



**Сложность алгоритма**

**Время:**
- Средний случай: O(N log N)
- Худший случай: O(N²) (при неудачном выборе pivot)
- Лучший случай: O(N log N)

**Память:**
- O(N) для хранения входного вектора
- O(N) для временного буфера при слиянии
- O(log N) для явного стека

## 4. Схема распараллеливания

Параллельная реализация использует распределение данных между процессами и последующим сбором результатов с эффективным многосторонним слиянием.

**Общая схема работы MPI версии**

```cpp
void ParallelQuickSort() {
  // 1. Распределение данных по процессам
  std::vector<int> local_buffer = DistributeData();

  // 2. Локальная сортировка на каждом процессе
  QuickSortWithSimpleMerge(local_buffer);

  // 3. Сбор размеров и данных на процессе 0
  MPI_Gather + MPI_Gatherv

  // 4. Слияние отсортированных частей на процессе 0
  //    - Итеративное слияние пар соседних отсортированных сегментов
  //    - Сложность O(N) вместо O(N log N) для полной пересортировки
  MultiWayMerge();

  // 5. Рассылка результата всем процессам
  MPI_Bcast
}
```

**Распределение данных**

```cpp
block_size = vector_size / proc_count      // Базовый размер блока
remainder = vector_size % proc_count       // Остаточные элементы
```

**Распределение работы**

- Данные равномерно распределяются между процессами
- Процессы с меньшим рангом получают дополнительный элемент при наличии остатка
- Используется MPI_Scatterv для неравномерного распределения

**Роли процессов**

- **Процесс 0:** Хранит исходные данные, выполняет распределение данных, собирает результаты, выполняет финальную сортировку.
- **Все процессы:** Выполняют локальную сортировку, отправляют результаты на процесс 0.


**Схема коммуникации**
- **Фаза вычислений:** 
    - MPI_Bcast - рассылка размера данных
    - MPI_Scatterv - распределение данных по процессам
- **Фаза вычислений:** Независимая локальная сортировка на каждом процессе
- **Фаза сбора:**
  - MPI_Gather - сбор размеров локальных массивов
  - MPI_Gatherv - сбор отсортированных данных
- **Фаза синхронизации:** 
  - MPI_Bcast - рассылка финального результата всем процессам

## 5. Детали реализации

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциоанльные тесты
- `tests/performance/main.cpp` - тесты производительности

**Ключевые классы:**
- `SinevAQuicksortWithSimpleMergeSEQ` - последовательная реализация
- `SinevAQuicksortWithSimpleMergeMPI` - параллельная MPI реализация

**Основные методы:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - подготовительные вычисления  
- `RunImpl()` - основной алгоритм
- `PostProcessingImpl()` - завершающая обработка
- `Partition(), SimpleMerge(), QuickSortWithSimpleMerge() ParallelQuickSort()` - вспомогательные методы

**Особенности реализации MPI версии:**

```cpp
bool SinevAQuicksortWithSimpleMergeMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Проверка выполняется только на процессе 0
  if (rank == 0) {
    if (GetInput().empty()) {
      return false;
    }
  }

  return true;
}

bool SinevAQuicksortWithSimpleMergeMPI::RunImpl() {
  ParallelQuickSort();
  return true;
}
```


**Структура данных для распределения**

```cpp
struct DistributionInfo {
  int local_size;
  std::vector<int> send_counts;
  std::vector<int> displacements;
};

static DistributionInfo PrepareDistributionInfo(int total_size, 
                                               int world_size, 
                                               int world_rank) {
  DistributionInfo info;
  
  int base_size = total_size / world_size;
  int remainder = total_size % world_size;
  
  info.local_size = base_size + (world_rank < remainder ? 1 : 0);
  info.send_counts.assign(world_size, 0);
  info.displacements.assign(world_size, 0);
  
  if (world_rank == 0) {
    int displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      info.send_counts[i] = base_size + (i < remainder ? 1 : 0);
      info.displacements[i] = displacement;
      displacement += info.send_counts[i];
    }
  }
  
  return info;
}
```


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
PPC_NUM_PROC=1,2,4,5,6,7,8,16
```

## 7. Результаты

### 7.1. Корректность работы

Все 14 функциональных тестов пройдены успешно:
- basic_case: Базовый случай [5, 3, 8, 1, 9, 2]
- with_negatives: С отрицательными числами [10, -5, 7, 0, -15, 3]
- already_sorted: Уже отсортированный массив [1, 2, 3, 4, 5, 6]
- reverse_sorted: Обратно отсортированный [6, 5, 4, 3, 2, 1]
- with_duplicates:С повторяющимися элементами [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
- all_equal: Все элементы одинаковые [7, 7, 7, 7, 7]
- single_element: Один элемент [42]
- two_sorted: Два отсортированных элемента [1, 2]
- two_unsorted: Два неотсортированных элемента [2, 1]
- even_size_large: Четный размер [9, 3, 7, 1, 8, 2, 6, 4, 5, 0]
- odd_size_large: Нечетный размер [11, 3, 9, 1, 7, 5, 8, 2, 6, 4, 10]
- with_zeros: С нулевыми значениями [0, -1, 0, 1, -2, 2]
- extreme_values: Экстремальные значения [2147483647, -2147483648, 0, 100, -100]
- mixed_case: Смешанный случай [100, -50, 0, 25, -25, 75, -75, 50, -100]

SEQ и MPI версии выдают идентичные результаты для всех тестовых случаев.

### 7.2. Производительность

**Время выполнения (секунды):**

| Версия | Количество процессов | Task Run время | Pipeline время |
|--------|---------------------|-----------------|----------------|
| SEQ    | 1                   | 0.00332          | 0.01857         |
| MPI    | 1                   | 0.00332          | 0.01857         |
| MPI    | 2                   | 0.00179          | 0.01399         |
| MPI    | 4                   | 0.00184          | 0.01138         |
| MPI    | 7                   | 0.00376          | 0.01090         |
| MPI    | 8                   | 0.00522          | 0.01162         |
| MPI    | 16                  | 0.00347          | 0.01195         |

### 7.3. Анализ эффективности

**Ускорение относительно SEQ версии:**

| Количество процессов | Ускорение | Эффективность |
|---------------------|-----------|---------------|
| 1                   | 0.97×     | 97%          |
| 2                   | 0.86x     | 93%           |
| 4                   | 0.81x     | 45%           |
| 7                   | 0.88x     | 13%           |
| 8                   | 0.64x     | 8%           |
| 16                  | 0.96x     | 6%           |



**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%


### 7.4. Наблюдения

1. **MPI с 1 процессом:** Производительность близка к SEQ версии (97% эффективности), небольшие накладные расходы MPI
2. **MPI с 2 процессами:** Наилучшее ускорение (1.86×) с высокой эффективностью (93%) - оптимальная конфигурация для данного объема данных
3. **MPI с 4 процессами:** Хорошее ускорение (1.81×), но снижение эффективности до 45% из-за увеличения накладных расходов
4. **MPI с 7+ процессами:** Резкое падение производительности из-за:
  - Высоких коммуникационных затрат
5. **Оптимальная конфигурация:** 2 процесса показывают наилучшее абсолютное время (0.00178512 сек) и максимальное ускорение


## 8. Выводы

### 8.1. Достигнутые результаты

- **Корректность:** Успешное прохождение всех функциональных тестов
- **Работоспособность:** Алгоритм корректно работает для всех граничных случаев
- **Универсальность:** Обработка различных типов данных (отрицательные, дубликаты, экстремальные значения)
- **Оптимизация:** Внедрение многостороннего слияния вместо полной пересортировки на процессе 0

### 8.2. Особенности реализации

- **Гибридный подход:** Комбинация быстрой сортировки и простого слияния
- **Итеративная реализация:** Использование явного стека вместо рекурсии для надежности
- **Балансировка нагрузки:** MPI_Bcast для согласованной проверки входных данных
- **Оптимизация коммуникаций:** 
  - Использование MPI_Scatterv для неравномерного распределения
  - MPI_Gatherv для сбора данных разного размера
  - MPI_Bcast для рассылки результатов

### 8.3. Ограничения и перспективы
- **Текущие ограничения:** 
  - Высокие коммуникационные затраты при малом количестве данных на процесс
  - Последовательное слияние на процессе 0 может стать узким местом для большого числа процессов
  - Статическое распределение данных без учета вычислительной мощности узлов
- **Перспективы:**  Использование различных алгоритмов сортировки в зависимости от характеристик данных

## 9. Источники
1. Лекции по параллельному программированию Сысоева А. В
2. Документация MPI: https://www.open-mpi.org/
3. Материалы курса: https://github.com/learning-process/ppc-2025-processes-engineers

## 10. Приложение

```cpp
#include "sinev_a_quicksort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <stack>
#include <algorithm>
#include <vector>

#include "sinev_a_quicksort_with_simple_merge/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace sinev_a_quicksort_with_simple_merge {

SinevAQuicksortWithSimpleMergeMPI::SinevAQuicksortWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool SinevAQuicksortWithSimpleMergeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (GetInput().empty()) {
      return false;
    }
  }

  return true;
}

bool SinevAQuicksortWithSimpleMergeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
  }
  return true;
}

int SinevAQuicksortWithSimpleMergeMPI::Partition(std::vector<int> &arr, int left, int right) {
  int pivot_index = left + ((right - left) / 2);
  int pivot_value = arr[pivot_index];

  std::swap(arr[pivot_index], arr[left]);

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) {
      i++;
    }

    while (i <= j && arr[j] > pivot_value) {
      j--;
    }

    if (i < j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    } else {
      break;
    }
  }

  std::swap(arr[left], arr[j]);
  return j;
}

void SinevAQuicksortWithSimpleMergeMPI::SimpleMerge(std::vector<int> &arr, int left, int mid, int right) { 
  if (left >= right) {
    return;
  }

  std::vector<int> temp(right - left + 1);

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }

  while (j <= right) {
    temp[k++] = arr[j++];
  }

  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void SinevAQuicksortWithSimpleMergeMPI::QuickSortWithSimpleMerge(std::vector<int> &arr, int left, int right) {
  struct Range {
    int left;
    int right;
  };
  
  std::stack<Range> stack;
  stack.push({left, right});
  
  while (!stack.empty()) {
    Range current = stack.top();
    stack.pop();
    
    int l = current.left;
    int r = current.right;
    
    if (l >= r) {
      continue;
    }

    int pivot_index = Partition(arr, l, r);
    
    int left_size = pivot_index - l;
    int right_size = r - pivot_index;
    
    if (left_size > 1 && right_size > 1) {
      if (left_size > right_size) {
        stack.push({pivot_index + 1, r});  
        stack.push({l, pivot_index - 1});  
      } else {
        stack.push({l, pivot_index - 1}); 
        stack.push({pivot_index + 1, r});  
      }
    } else if (left_size > 1) {
      stack.push({l, pivot_index - 1});
    } else if (right_size > 1) {
      stack.push({pivot_index + 1, r});
    }
    
    SimpleMerge(arr, l, pivot_index, r);
  }
}

SinevAQuicksortWithSimpleMergeMPI::DistributionInfo SinevAQuicksortWithSimpleMergeMPI::PrepareDistributionInfo(
    int total_size, int world_size, int world_rank) {
  DistributionInfo info;

  int base_size = total_size / world_size;
  int remainder = total_size % world_size;

  info.local_size = base_size + (world_rank < remainder ? 1 : 0);

  info.send_counts.assign(world_size, 0);
  info.displacements.assign(world_size, 0);

  if (world_rank == 0) {
    int displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      info.send_counts[i] = base_size + (i < remainder ? 1 : 0);
      info.displacements[i] = displacement;
      displacement += info.send_counts[i];
    }
  }

  return info;
}

std::vector<int> SinevAQuicksortWithSimpleMergeMPI::DistributeData(int world_size, int world_rank) {
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  DistributionInfo info = PrepareDistributionInfo(total_size, world_size, world_rank);

  std::vector<int> local_buffer(info.local_size);

  MPI_Scatterv(GetOutput().data(), info.send_counts.data(), info.displacements.data(), MPI_INT, local_buffer.data(),
               info.local_size, MPI_INT, 0, MPI_COMM_WORLD);

  return local_buffer;
}

void SinevAQuicksortWithSimpleMergeMPI::ParallelQuickSort() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size == 1) {
    if (!GetOutput().empty()) {
      QuickSortWithSimpleMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
    }
    return;
  }

  // 1. Распределение данных по процессам
  std::vector<int> local_buffer = DistributeData(world_size, world_rank);

  // 2. Локальная сортировка на каждом процессе
  if (!local_buffer.empty()) {
    QuickSortWithSimpleMerge(local_buffer, 0, static_cast<int>(local_buffer.size()) - 1);
  }

  // 3. Сбор размеров от всех процессов
  std::vector<int> all_sizes(world_size);
  int local_size = static_cast<int>(local_buffer.size());

  MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 4. Подготовка буфера на корневом процессе
  std::vector<int> displacements(world_size, 0);
  int total_size = 0;

  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      displacements[i] = total_size;
      total_size += all_sizes[i];
    }
    GetOutput().resize(total_size);
  }

  // 5. Сбор данных на корневом процессе
  int *recv_data = (world_rank == 0) ? GetOutput().data() : nullptr;
  int *recv_counts = (world_rank == 0) ? all_sizes.data() : nullptr;
  int *recv_displs = (world_rank == 0) ? displacements.data() : nullptr;

  MPI_Gatherv(local_buffer.data(), local_size, MPI_INT, recv_data, recv_counts, recv_displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  // 6. Многостороннее слияние отсортированных частей на процессе 0
  if (world_rank == 0 && !GetOutput().empty()) {
    std::vector<std::pair<int, int>> segments;
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      if (all_sizes[i] > 0) {
        segments.emplace_back(offset, offset + all_sizes[i] - 1);
        offset += all_sizes[i];
      }
    }

    while (segments.size() > 1) {
      std::vector<std::pair<int, int>> next_segments;
      for (size_t i = 0; i < segments.size(); i += 2) {
        if (i + 1 < segments.size()) {
          int left = segments[i].first;
          int mid = segments[i].second;
          int right = segments[i + 1].second;
          SimpleMerge(GetOutput(), left, mid, right);
          next_segments.emplace_back(left, right);
        } else {
          next_segments.push_back(segments[i]);
        }
      }
      segments = std::move(next_segments);
    }
  }

  // 7. Рассылка результата (оставляем для корректной работы тестов)
  int final_size = 0;
  if (world_rank == 0) {
    final_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank != 0) {
    GetOutput().resize(final_size);
  }

  MPI_Bcast(GetOutput().data(), final_size, MPI_INT, 0, MPI_COMM_WORLD);
}

bool SinevAQuicksortWithSimpleMergeMPI::RunImpl() {
  ParallelQuickSort();
  return true;
}

bool SinevAQuicksortWithSimpleMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sinev_a_quicksort_with_simple_merge

```