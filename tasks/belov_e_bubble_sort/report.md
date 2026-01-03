# Сортировка пузырьком (алгоритм чет-нечетной перестановки)

- **Студент**: Белов Егор Алексеевич, группа 3823Б1ПР2
- **Технология**: SEQ | MPI
- **Вариант**: 21

## 1. Введение
Сортировка массива данных является важнейшией и фундаментальной операцией. Она крайне необходима для хранения и анализа данных. Одним из алгоритмов сортировки является сортировкой пузырьком. Этот алгоритм выполняется последовательно одним процессором.

Сортировку пузырьком можно ускорить через параллельный алгоритм чет-нечетной перестановки (или слияния для подмассивов данных).

Наша задача разработать параллельный алгоритм сортировки пузырьком.

## 2. Постановка задачи
Дан массив данных длинной n.

Требуется вернуть отсортированный массив данных.

Тип входных данных:
```cpp
using InType = std::vector<int>;
```

Тип выходных данных:
```cpp
using OutType = std::vector<int>;
```

## 3. Базовый алгоритм (Sequential)
Алгоритм последовательно двигает максимальный элемент массива в конец массива и уменьшает видимость массив на один элемент, который записывается в выходный массив.
```cpp
  for (size_t i = 0; i < n; out[n - i - 1] = data[n - i - 1], i++) {
    for (size_t j = 0; j < n - i - 1; j++) {
      if (data[j] > data[j + 1]) {
        temp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = temp;
      }
    }
  }
```

## 4. Схема распараллеливания
### Краткое описание
1. Возьмём массив длинной n
2. Распределим n/p элементов из массива с остатком на p процессов в локальные массивы, где p - общее количество процессов.
3. Отсортируем на каждом процессе их локальный массив пузырьковой сортировкой.
4. Сделаем p + 1 шагов. На каждом шаге процессы будут упорядоченно обмениваться своими подмассивами. Если шаг чётный, то будут обмениваться процессы начиная с нулевой позиции со своими правыми соседями, если шаг нечётный, то будут обмениваться процессы начиная с первой позиции со своими правыми соседями.
5. Собираем все локальные массивы с процессов в один финальный отсортированный массив.

### Алгоритм распределения массива с нулевого процесса на локальные подмассивы всех процессов
```cpp
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

int local_arr_size = 0;
std::vector<int> local_arr;
int base = n / mpi_size;
int rem = n % mpi_size;
local_arr_size = base + (rank < rem ? 1 : 0);
local_arr.resize(local_arr_size);

std::vector<int> arrays_sizes;
arrays_sizes.resize(mpi_size);
MPI_Allgather(&local_arr_size, 1, MPI_INT, arrays_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

std::vector<int> displs;
displs.resize(mpi_size);
if (mpi_size == 0) {
  return false;
}
displs[0] = 0;
for (int i = 1; i < mpi_size; i++) {
  displs[i] = displs[i - 1] + arrays_sizes[i - 1];
}

MPI_Scatterv(arr.data(), arrays_sizes.data(), displs.data(), MPI_INT, local_arr.data(), local_arr_size, MPI_INT, 0,
             MPI_COMM_WORLD);
```

### Алгоритм пузырьковой сортировки подмассива на процессе
```cpp
void BubbleSort(std::vector<int> &arr) {
  const std::size_t n = arr.size();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j + 1 < n - i; ++j) {
      if (arr[j] > arr[j + 1]) {
        std::swap(arr[j], arr[j + 1]);
      }
    }
  }
}
```

### Алгоритмы отсортированного обмена подмассивами между соседями
```cpp
std::vector<int> LeftMerge(const std::vector<int> &left, const std::vector<int> &right) {
  std::vector<int> result;
  result.reserve(left.size() + right.size());

  std::ranges::merge(left, right, std::back_inserter(result));

  return {result.begin(), std::next(result.begin(), static_cast<std::vector<int>::difference_type>(left.size()))};
}
std::vector<int> RightMerge(const std::vector<int> &left, const std::vector<int> &right) {
  std::vector<int> result;
  result.reserve(left.size() + right.size());

  std::ranges::merge(left, right, std::back_inserter(result));

  return {std::prev(result.end(), static_cast<std::vector<int>::difference_type>(right.size())), result.end()};
}

void LeftProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
                 MPI_Comm comm) {
  std::vector<int> right_arr;
  right_arr.resize(arrays_sizes[rank + 1]);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank + 1, 0, right_arr.data(), arrays_sizes[rank + 1],
               MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = LeftMerge(local_arr, right_arr);
}

void RightProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
                  MPI_Comm comm) {
  std::vector<int> left_arr;
  left_arr.resize(arrays_sizes[rank - 1]);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank - 1, 0, left_arr.data(), arrays_sizes[rank - 1], MPI_INT,
               rank - 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = RightMerge(left_arr, local_arr);
}
```

### Алгоритмы распределения обменов на чет-нечетных фазах
```cpp
void EvenPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
               MPI_Comm comm) {
  if (mpi_size % 2 != 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    LeftProcAct(rank, local_arr, local_arr_size, arrays_sizes, comm);
  } else {
    RightProcAct(rank, local_arr, local_arr_size, arrays_sizes, comm);
  }
}

void OddPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, std::vector<int> &arrays_sizes,
              MPI_Comm comm) {
  if (rank == 0) {
    return;
  }
  if (mpi_size % 2 == 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    RightProcAct(rank, local_arr, local_arr_size, arrays_sizes, comm);
  } else {
    LeftProcAct(rank, local_arr, local_arr_size, arrays_sizes, comm);
  }
}
```

### Алгоритм запуска шагов
```cpp
for (int i = 0; i < mpi_size + 1; i++) {
    if (i % 2 == 0) {
        EvenPhase(rank, mpi_size, local_arr, local_arr_size, arrays_sizes, MPI_COMM_WORLD);
    } else {
        OddPhase(rank, mpi_size, local_arr, local_arr_size, arrays_sizes, MPI_COMM_WORLD);
    }
}
```

### Алгоритм сбора локальных подмассивов в один массив
```cpp
arr.resize(n);
MPI_Allgatherv(local_arr.data(), local_arr_size, MPI_INT, arr.data(), arrays_sizes.data(), displs.data(), MPI_INT,
                MPI_COMM_WORLD);
```

### Cхема параллельной работы алгоритма
1. Каждый процесс определяет количество процессов и свой ранг и записывает соответственно в переменные **mpi_size** и **rank**.
2. Нулевой процесс принимает входные данные из **GetInput()**, расчитывает размер **n** и транслирует каждому процессу его через **MPI_Bcast()**.
3. Каждый процесс расчитывает размер своего подмассива **local_arr_size** через **base + (rank < rem ? 1 : 0)**, где **base = n / mpi_size** и **rem = n % mpi_size**.
4. Каждый процесс отсылает свой **local_arr_size** в **MPI_Allgather()** и запоминает размер подмассива каждого процесса в массиве **arrays_sizes**.
5. Каждый процесс рассчитывает массив **displs**, необходимый для **MPI_Scatterv()**.
6. На каждый процесс распределяется свой подмассив через **MPI_Scatterv()** размером **local_arr_size**.
7. Каждый процесс сортирует свой подмассив через **BubbleSort()**.
8. На каждом процессе начинается цикл чет-нечетных перестановок с **mpi_size + 1** шагами. На четную фазу процессы запускают **EvenPhase()**, на нечётную - **OddPhase()**. Внутри фаз процессы делятся на пары, где **i** процессу ставится в пару **i + 1** процесс, на чётной фазе **i = 0**, на нечётной **i = 1**. Левый и правый процесс из пары соответственно запускают **LeftProcAct()** и **RightProcAct()**. Процессы в паре запускают **MPI_Sendrecv()** отправляя свой подмассив и получая подмассив соседа, а затем левый и правый процесс из пары запускают упорядоченную перестановку через **LeftMerge()** и **RightMerge()** и записывают в свой подмассив данные из своей части.
9. Все процессы собирают из своих подмассивов финальный отсортированный массив через **MPI_Allgatherv()** и записывают ответ в **GetOutput()**.
## 5. Детали реализации
|          Файл          |                 Назначение                  |
|------------------------|---------------------------------------------|
| `common.hpp`           | Определение входных, выходных типов задачи и тип тестов |
| `ops_seq.hpp/.cpp`     |         Последовательная реализаци  |
| `ops_mpi.hpp/.cpp`     |                MPI-реализация         |
| `functional/main.cpp`  |             Функциональные тесты          |
| `performance/main.cpp` |       Тесты производительности       |
## 6. Экспериментальная среда
|  Компонент |               Значение                       |
|------------|----------------------------------------------|
|     CPU    |           Apple M1                 |
|     RAM    |                 8 GB                       |
|     ОС     | OS: Ubuntu 24.04 (DevContainer / macOs 26.1) |
| Компилятор | GCC 13.3.0 (g++), C++20, CMake, Release     |
|     MPI    |        mpirun (Open MPI) 4.1.6            |
## 7. Результаты и обсуждение
### 7.1 Корректность
Для функциональных тестов: 3 .txt файла, содержащие перечисление чисел через пробел.

Тесты были подобраны для проверок всех уникальных случаев работы алгоритма. Все тесты прошли проверку на SEQ и MPI реализации.
### 7.2 Производительность
Для тестов производительности: генерируется убывающий массив из 20000 элементов
| Mode        | Count | Time, s  | Speedup | Efficiency |
|-------------|-------|----------|---------|------------|
| seq         | 1     | 0.366000 | 1.00    | N/A        |
| mpi         | 1     | 0.361000 | 1.01    | 101.4%     |
| mpi         | 2     | 0.103000 | 3.55    | 177.5%     |
| mpi         | 4     | 0.037000 | 9.89    | 247.25%    |
| mpi         | 8     | 0.018000 | 20.33   | 254.17%    |
## 8. Заключение
В ходе выполнения работы удалось реализовать алгоритм пузырьковой сортировки, распараллелить его при помощи MPI и метода чет-нечетных перестановок и увидеть эффективность работы параллельного алгоритма.
## 9. Источники
1. Сысоев А. В. Курс лекций по параллельному программированию
2. Документация Open MPI     https://www.open-mpi.org/doc/
3. Microsoft Функции MPI     https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions