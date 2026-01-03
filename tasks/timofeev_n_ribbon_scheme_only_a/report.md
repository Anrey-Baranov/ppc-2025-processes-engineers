#  Ленточная горизонтальная схема - разбиение только матрицы А - умножение матрицы на матрицу

- Студент: Тимофеев Никита Сергеевич, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 13

## 1. Введение
Применение технологии MPI и сравнение производительности программ с использованием MPI и последовательным выполнением, на примере задачи ленточной горизонтальной схемы перемножения матриц с разбиением матрицы А.

## 2. Постановка задачи
Задача: на вход программы подаются две матрицы, необходимо реализовать ленточную горизонтальную схему перемножения матриц с разбиением матрицы А.
Формат входных данных: ```std::pair<MatrixType, MatrixType>```
Формат выходных данных: ```MatrixType```
```MatrixType``` раскрывается как ```std::vector<std::vector<int>>```
Необходимо создать две программы, одна из которых использует последовательное выполнение, другая - использует преимущества технологии MPI, затем сравнить их скорость.

## 3. Базовый алгоритм (Последовательный)
```cpp
bool TimofeevNRibbonSchemeOnlyASEQ::RunImpl() {
  MatrixType a = GetInput().first;
  MatrixType b = GetInput().second;
  std::vector<std::vector<int>> c(a.size(), std::vector<int>(b[0].size()));
  for (size_t i = 0; i < a.size(); i++) {
    for (size_t j = 0; j < b[0].size(); j++) {
      c[i][j] = CalculatingCElement(a, b, i, j);
    }
  }
  GetOutput() = c;
  return true;
}
```

## 4. Схема распараллеливания
- Для MPI: распараллеливание ведётся следующим образом: нулевой процесс раздаёт данные остальным по такому принципу: выбирается k = (n / (количество процессов - 1)) + (n % (количество процессов - 1) > 0 ? 1 : 0) - иными словами, k >= n / (количество процессов - 1), и всегда положительно. Процессам, начиная с 1-го, рассылается по k строк матрицы А, если на каком-то шаге осталось d != k строк, они отсылаются последнему процессу. Далее процессы вычисляют свою часть матрицы С (присланное количество строк А * количество столбцов B) и отсылают её нулевому процессу, который собирает все части в единый результат и рассылает всем остальным.

## 5. Детали реализации
- Структура кода:
Есть классы ```TimofeevNRibbonSchemeOnlyAMPI``` и ```TimofeevNRibbonSchemeOnlyASEQ```, там функция ```bool RunImpl()```. В последовательной программе реализован алгоритм, перемножающий матрицы А и B тройным вложенным циклом, вычисляя каждый элемент матрицы C[i][j], равный сумме произведений A[i][t] * B[t][j] для всех t от 1 до количества строк в матрице B, если A - матрица n на m, B - матрица m на p, то сложность - O(n * p * m) (в случае квадратных матриц - O(n^3)). Алгоритм для параллельной программы описан выше (Раздел 4. Схема распараллеливания), процессы, которым рассылаются строки A и матрица B вычисляют элементы матрицы C аналогично тому, как это происходит в программе для последовательного исполнения (просто количество строк в этом случае меньше), и если мы возьмём s - число строк матрицы A, которое отсылается последнему процессу в случае, если n оказалось не кратно k (т. е. последний процесс получает либо больше, либо меньше строк A), то сложность составляет O(max{[n / k], s} * p * m). Код для функции параллельного исполнения приведён в Приложении 1.
- Реализация предполагает, что если произошёл запуск с 1 процессом, всю работу выполняет единственный (нулевой) процесс.

## 6. Экспериментальная установка
- Аппаратное обеспечение / ОС: Intel(R) Core(TM) i7-3770, 8 ядер / 1 поток на ядро, 23.4 ГиБ ОЗУ, ОС: кубунту 24.04
- Инструментальная цепочка: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0, clang-tidy-21, clang-format-21, тип сборки: Release
- Данные: тестовые данные для функциональных тестов находятся в файле tests/functional/main.cpp в виде списка типа ``` std::array<TestType, 16>```, тестовые данные для тестирования производительности представляют из себя пару матриц 1000 на 1000, причём матрица А единичная, а B состоит из одних только единиц.

## 7. Результаты и выводы

### 7.1 Корректность
Корректность была проверена с помощью функциональных тестов. Обе программы корректно перемножают квадратные матрицы одинакового размера, одномерные векторы размеров 1 на n и n на 1 соответственно, матрицы размеров n на m и m на k соответственно (все тестовые данные есть в tests/functional/main.cpp)

### 7.2 Производительность
Время предоставлено для запусков с 1, 2, 3, 4, 7 и 8 процессами соответственно
| Mode         | Time, s        | Speedup | ProcNum |
|--------------|----------------|---------|---------|
| seq, task_run| 6.6289003849   | 1.00    |    1    |
| mpi, task_run| 6.5665170576   | 1.01    |    1    |
| seq, task_run| 6.7115401745   | 1.00    |    2    |
| mpi, task_run| 6.5540123372   | 1.02    |    2    |
| seq, task_run| 6.7019720554   | 1.00    |    3    |
| mpi, task_run| 3.3579918812   | 2.00    |    3    |
| seq, task_run| 6.8010344982   | 1.00    |    4    |
| mpi, task_run| 2.2836519102   | 3.00    |    4    |
| seq, task_run| 6.7019720545   | 1.00    |    7    |
| mpi, task_run| 0.9787079615   | 6.88    |    7    |
| seq, task_run| 6.7019720443   | 1.00    |    8    |
| mpi, task_run| 0.8563694663   | 7.83    |    8    |
Speedup для mpi = (seq time / mpi time)

## 8. Заключение
Были написаны две программы для сарвнения последовательного выполнения и mpi решения задачи ленточной горизонтальной схемы перемножения матриц с разбиением матрицы А. В результате применения технологии MPI программа ускорилась.

## Источники

1. Ленточные алгоритмы переумножения матриц - www.hpcc.unn.ru. Ссылка: http://www.hpcc.unn.ru/?dir=808

## Приложение
Приложение 1. Код функции параллельного исполнения
```cpp
bool TimofeevNRibbonSchemeOnlyAMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1) {
    MatrixType a = GetInput().first;
    MatrixType b = GetInput().second;
    GetOutput().resize(a.size());
    CalculatingCOneProcess(a, b, GetOutput());
    return true;
  }

  if (rank == 0) {
    MatrixType a = GetInput().first;
    MatrixType b = GetInput().second;
    size_t k = 0;
    k = (a.size() / static_cast<size_t>(size - 1)) + (a.size() % static_cast<size_t>(size - 1) > 0 ? 1 : 0);
    size_t a_size = a.size();
    size_t a_row_size = a[0].size();
    size_t b_size = b.size();
    size_t b_row_size = b[0].size();
    BroadcastingParameters(a_size, a_row_size, b_size, b_row_size);
    BroadcastingB(b);
    SendingAParts(a, size, k);
    std::vector<std::vector<int>> cmatr(a.size(), std::vector<int>(b_row_size));
    ReceivingCParts(cmatr, size, k, b_row_size);
    for (auto &i : cmatr) {  // рассылка того, что получилось
      MPI_Bcast(i.data(), static_cast<int>(b_row_size), MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = cmatr;
  } else {
    size_t k = 0;
    size_t a_size = 0;
    size_t a_row_size = 0;
    size_t b_size = 0;
    size_t b_row_size = 0;
    BroadcastingParameters(a_size, a_row_size, b_size, b_row_size);
    std::vector<std::vector<int>> b_copy(b_size, std::vector<int>(b_row_size));
    BroadcastingB(b_copy);
    MPI_Recv(&k, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<std::vector<int>> a_part(k, std::vector<int>(a_row_size));
    for (size_t j = 0; j < k; j++) {  // принимаем строки A
      MPI_Recv(a_part[j].data(), static_cast<int>(a_row_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::vector<std::vector<int>> c_part(k, std::vector<int>(b_row_size));
    CalculatingCPart(k, a_part, b_copy, c_part);  // можем вычислить k строк (k * b_row_size элементов) матрицы C
    for (size_t i = 0; i < k; i++) {              // у нас есть матрица k * b_row_size
      MPI_Send(c_part[i].data(), static_cast<int>(b_row_size), MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    GetOutput().resize(a_size);
    for (size_t i = 0; i < a_size; i++) {  // принимает результат из Bcast
      GetOutput()[i].resize(b_row_size);
      MPI_Bcast(GetOutput()[i].data(), static_cast<int>(b_row_size), MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}
```
