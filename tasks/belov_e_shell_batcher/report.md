# Сортировка Шелла с четно-нечетным слиянием Бэтчера

- **Студент**: Белов Егор Алексеевич, группа 3823Б1ПР2
- **Технология**: SEQ | MPI
- **Вариант**: 17

## 1. Введение
Сортировка массива данных является важнейшией и фундаментальной операцией. Она крайне необходима для хранения и анализа данных. Одним из алгоритмов сортировки является сортировка Шелла. Этот алгоритм выполняется последовательно одним процессором.

Сортировку Шелла можно ускорить через параллельное четно-нечетное слияние Бэтчера.

Наша задача разработать параллельный алгоритм сортировки Шелла через четно-нечетное слияние Бэтчера.

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
Алгоритм последовательно сравнивает и сортирует элементы массива на некотором расстоянии gap, после каждой такой сортировки gap уменьшается пока не достигнет единицы, что гарантирует в конце работы алгоритма сортировку вставками.
```cpp
  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = data[i];
      size_t j = i;

      while (j >= gap && data[j - gap] > temp) {
        data[j] = data[j - gap];
        j -= gap;
      }

      data[j] = temp;
    }
  }
```

## 4. Схема распараллеливания
### Краткое описание
1. Возьмем массив длинной n.
2. Добавим в него такое количество мусорных максимальных данных garbage, чтобы впоследствии на каждом процессе размер подмассива был одинаковый.
3. Распределяем на каждый процесс подмассив длинной (n + garbage) / mpi_size.
4. Выполняем на каждом процессе сортировку Шелла локального подмассива.
5. Сделаем mpi_size + 1 шагов. На каждом шаге процессы будут упорядоченно обмениваться своими подмассивами. Если шаг чётный, то будут обмениваться процессы начиная с нулевой позиции со своими правыми соседями, если шаг нечётный, то будут обмениваться процессы начиная с первой позиции со своими правыми соседями. Каждое слияние будет происходить методом четных-нечетных слияний Бэтчера.
6. Собираем все локальные массивы с процессов в один финальный отсортированный массив.

### Алгоритм распределния массива на процессы
```cpp
  int mpi_size = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> arr;
  int n = 0;
  if (rank == 0) {
    arr = GetInput();
    n = static_cast<int>(arr.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int garbage = (n % mpi_size == 0) ? 0 : (mpi_size - (n % mpi_size));

  if (rank == 0) {
    for (int i = 0; i < garbage; i++) {
      arr.push_back(std::numeric_limits<int>::max());
    }
  }

  int local_arr_size = (n + garbage) / mpi_size;
  std::vector<int> local_arr;
  local_arr.resize(local_arr_size);

  MPI_Scatter(arr.data(), local_arr_size, MPI_INT, local_arr.data(), local_arr_size, MPI_INT, 0, MPI_COMM_WORLD);
```
### Алгоритм упорядочивание подмассива сортировкой Шелла
```cpp
void ShellSort(std::vector<int> &arr) {
  size_t n = arr.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = arr[i];
      size_t j = i;

      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }

      arr[j] = temp;
    }
  }
}
```
### Алгоритм проведения четных-нечетных слияний между процессами
```cpp
void LeftProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  std::vector<int> right_arr;
  right_arr.resize(local_arr_size);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank + 1, 0, right_arr.data(), local_arr_size, MPI_INT,
               rank + 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = BatcherLeftMerge(local_arr, right_arr, local_arr_size, rank, comm);
}

void RightProcAct(int rank, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  std::vector<int> left_arr;
  left_arr.resize(local_arr_size);
  MPI_Sendrecv(local_arr.data(), local_arr_size, MPI_INT, rank - 1, 0, left_arr.data(), local_arr_size, MPI_INT,
               rank - 1, 0, comm, MPI_STATUS_IGNORE);
  local_arr = BatcherRightMerge(left_arr, local_arr, local_arr_size, rank, comm);
}

void EvenPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  if (mpi_size % 2 != 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    LeftProcAct(rank, local_arr, local_arr_size, comm);
  } else {
    RightProcAct(rank, local_arr, local_arr_size, comm);
  }
}

void OddPhase(int rank, int mpi_size, std::vector<int> &local_arr, int local_arr_size, MPI_Comm comm) {
  if (rank == 0) {
    return;
  }
  if (mpi_size % 2 == 0 && rank == mpi_size - 1) {
    return;
  }
  if (rank % 2 == 0) {
    RightProcAct(rank, local_arr, local_arr_size, comm);
  } else {
    LeftProcAct(rank, local_arr, local_arr_size, comm);
  }
}
for (int i = 0; i < mpi_size + 1; i++) {
  if (i % 2 == 0) {
    EvenPhase(rank, mpi_size, local_arr, local_arr_size, MPI_COMM_WORLD);
  } else {
    OddPhase(rank, mpi_size, local_arr, local_arr_size, MPI_COMM_WORLD);
  }
}
```

### Алгоритм четного-нечетного слияния Бэтчера
```cpp
std::vector<int> BatcherLeftMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                  int local_arr_size, int rank, MPI_Comm comm) {
  std::vector<int> even_arr1;
  even_arr1.reserve(local_arr_size);
  std::vector<int> even_arr2;
  even_arr2.reserve(local_arr_size);
  for (int i = 0; i < local_arr_size; i += 2) {
    even_arr1.push_back(input_left_arr[i]);
  }
  for (int i = 0; i < local_arr_size; i += 2) {
    even_arr2.push_back(input_right_arr[i]);
  }
  std::vector<int> even_arr;
  even_arr.resize(even_arr1.size() + even_arr2.size());
  std::ranges::merge(even_arr1, even_arr2, even_arr.begin());

  std::vector<int> odd_arr;
  odd_arr.resize((2 * static_cast<size_t>(local_arr_size)) - even_arr.size());
  MPI_Sendrecv(even_arr.data(), static_cast<int>(even_arr.size()), MPI_INT, rank + 1, 0, odd_arr.data(),
               static_cast<int>(odd_arr.size()), MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);

  std::vector<int> left_arr;
  left_arr.reserve(local_arr_size);
  if (local_arr_size != 0) {
    left_arr.push_back(even_arr[0]);
  }
  int i = 0;
  while (left_arr.size() < static_cast<size_t>(local_arr_size)) {
    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    left_arr.push_back(comp.first);
    if (left_arr.size() < static_cast<size_t>(local_arr_size)) {
      left_arr.push_back(comp.second);
    }
    i++;
  }
  return left_arr;
}

std::vector<int> BatcherRightMerge(std::vector<int> &input_left_arr, std::vector<int> &input_right_arr,
                                   int local_arr_size, int rank, MPI_Comm comm) {
  std::vector<int> odd_arr1;
  odd_arr1.reserve(local_arr_size);
  std::vector<int> odd_arr2;
  odd_arr2.reserve(local_arr_size);
  for (int i = 1; i < local_arr_size; i += 2) {
    odd_arr1.push_back(input_left_arr[i]);
  }
  for (int i = 1; i < local_arr_size; i += 2) {
    odd_arr2.push_back(input_right_arr[i]);
  }

  std::vector<int> odd_arr;
  odd_arr.resize(odd_arr1.size() + odd_arr2.size());
  std::ranges::merge(odd_arr1, odd_arr2, odd_arr.begin());

  std::vector<int> even_arr;
  even_arr.resize((2 * static_cast<size_t>(local_arr_size)) - odd_arr.size());
  MPI_Sendrecv(odd_arr.data(), static_cast<int>(odd_arr.size()), MPI_INT, rank - 1, 0, even_arr.data(),
               static_cast<int>(even_arr.size()), MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);

  std::vector<int> right_arr;
  right_arr.reserve(local_arr_size);
  size_t i = 0;
  if (local_arr_size % 2 == 0) {
    i = (static_cast<size_t>(local_arr_size) / 2) - 1;

    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    right_arr.push_back(comp.second);
    i++;
  } else {
    i = (static_cast<size_t>(local_arr_size) - 1) / 2;
  }
  for (; i + 1 < even_arr.size() && i < odd_arr.size(); i++) {
    std::pair<int, int> comp = {even_arr[i + 1], odd_arr[i]};
    if (comp.first > comp.second) {
      comp = {comp.second, comp.first};
    }

    right_arr.push_back(comp.first);
    right_arr.push_back(comp.second);
  }
  for (; i + 1 < even_arr.size(); i++) {
    right_arr.push_back(even_arr[i + 1]);
  }
  for (; i < odd_arr.size(); i++) {
    right_arr.push_back(odd_arr[i]);
  }
  return right_arr;
}
```
### Cхема параллельной работы алгоритма
1. Каждый процесс определяет количество процессов и свой ранг и записывает соответственно в переменные **mpi_size** и **rank**.
2. Нулевой процесс записывает входные данные из **GetInput()** в **arr**, вычисляет размер массива **n** и рассылает его всем процессам через **MPI_Bcast()**.
3. Каждый процесс вычисляет количество мусорных данных **garbage**, которое нужно добавить к входному массиву, чтобы на всех процессах впоследствии были подмассивы одинаковой длинны.
4. Нулевой процесс добавляет мусорные данные в виде максмально возможного числа типа **int** в массив.
5. Каждый процесс вычисляет длину своего подмассива **local_arr_size** через формулу **local_arr_size** = (**n** + **garbage**) / **mpi_size**.
6. Нулевой процесс распределяет подмассивы **local_arr** из входных данных **arr** на каждый процесс через **MPI_Scatter()**.
7. Каждый процесс запускает сортировку Шелла **ShellSort()** над своим локальным массивом **local_arr**.
8. Каждый процесс запускает цикл четных-нечетных слияний длинной **mpi_size + 1**. На каждом шаге в зависимости от чётности шага процессы разбиваются на пары, в которых процессы обмениваются своими подмассивами с соседом через **MPI_Sendrecv()**. Далее процессы запускают алгоритм четных-нечетных слияний Бэтчера: левый сосед сливает все элементы из массивов соседей, стоящих на чётных местах, **even_arr**, а правый сосед на нечетных **odd_arr**. Затем левый и правый процесс в паре обмениваются **even_arr** и **odd_arr** через **MPI_Sendrecv()**. В конце процесс последовательно забирает элементы своей части с помощью компаратора пары элементов из **even_arr** и **odd_arr**, и получившийся массив становится **local_arr** этого процесса.
9. Каждый процесс отсылает свой локальный массив **local_arr** через **MPI_Allgather()** и получает финальный отсортированный массив **arr**, очищенный от мусорных данных.
10. Каждый процесс отправляет **arr** в **GetOutput()**.

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
Для тестов производительности: генерируется убывающий массив из 1000000 элементов
| Mode        | Count | Time, s  | Speedup | Efficiency |
|-------------|-------|----------|---------|------------|
| seq         | 1     | 0.100847 | 1.00    | N/A        |
| mpi         | 1     | 0.100645 | 1.00    | 100.20%    |
| mpi         | 2     | 0.054625 | 1.85    | 92.31%     |
| mpi         | 4     | 0.019064 | 5.29    | 132.19%    |
| mpi         | 8     | 0.008246 | 12.23   | 152.87%    |
## 8. Заключение
В ходе выполнения работы удалось реализовать алгоритм сортировки Шелла, распараллелить его при помощи MPI, метода чет-нечетных перестановок и чет-нечетного слияния Бэтчера и увидеть эффективность работы параллельного алгоритма.
## 9. Источники
1. Сысоев А. В. Курс лекций по параллельному программированию.
2. Документация Open MPI.     https://www.open-mpi.org/doc/
3. Microsoft Функции MPI.     https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions
4. Якобовский М.В. Параллельные алгоритмы сортировки больших объемов данных.     http://lira.imamod.ru/FondProgramm/Sort