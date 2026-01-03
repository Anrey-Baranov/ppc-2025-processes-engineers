- Student: <Бузулукский Данила Сергеевич>, group <3823Б1ПР5>
- Technology: <SEQ | MPI>
- Variant: <21>

## 1. Introduction
МОТИВАЦИЯ -> Ускорение сортировки массивов большого размера за счёт распределения нагрузки по нескольким процессам
ПРОБЛЕМА  ->  Алгоритм пузырьковой сортировки имеет квадратичную сложность, что делает его неэффективным для больших объёмов данных
РЕЗУЛЬТАТ -> Значительное ускорение сортировки за счёт параллельной обработки на нескольких процессах

## 2. Problem Statement
Formal task definition ->  Для массива целых чисел размера N выполнить сортировку по возрастанию
input/output format    -> На вход подаётся std::vector<int>, на выходе - отсортированный std::vector<int> той же длины
constraints            -> N >= 0, все элементы - целые числа

## 3. Baseline Algorithm (Sequential)
Describe the base algorithm with enough detail to reproduce.

Основной алгоритм сортировки
```cpp
  bool sorted = false;
  for (size_t pass = 0; pass < n && !sorted; ++pass) {
    sorted = true;

    for (size_t i = 0; i + 1 < n; i += 2) {
      if (arr[i] > arr[i + 1]) {
        std::swap(arr[i], arr[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i + 1 < n; i += 2) {
      if (arr[i] > arr[i + 1]) {
        std::swap(arr[i], arr[i + 1]);
        sorted = false;
      }
    }

    if (sorted) {
      break;
    }
  }
```

Про алгоритм:
 Алгоритм легко поддаётся паралелизму, алгоритм обладает стабильной работой,
 Алгоритм может эффективно показывать себя при большом объёме данных, но только при паралелизме(при слишком большом объёме данных проседает)


## 4. Parallelization Scheme
data distribution:
Блочное распределение массива между процессами
, communication pattern:
```cpp
MPI_Scatterv(...);         // распределение данных
LocalOddEvenSort(local);    // локальная сортировка
for (int phase = 0; phase < proc_count + 1; ++phase) {
    int partner = PartnerRank(rank, phase);
    ExchangeWithNeighbor(local, rank, partner, counts);
}
MPI_Gatherv(...);           // сбор результатов
MPI_Bcast(...);             // распространение финального результата

```
 rank roles:
    рабочие процессы (rank < N): получают данные, сортируют локально, участвуют в обменах и возвращают результат.
    процессы без данных (rank ≥ N): не участвуют в вычислениях, получают финальный результат через MPI_Bcast.
    Rank 0: распределяет данные, собирает результаты и рассылает финальный результат.


## 5. Implementation Details
- Code structure (files, key classes/functions)
common - общие структуры данных
mpi - паралельная реализация mpi
seq - последовательная
test - тесты функцианальности и производительности

```cpp
class BuzulukskyDBubbleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BuzulukskyDBubbleSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
```
```cpp
class BuzulukskyDBubbleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BuzulukskyDBubbleSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

```
```cpp
using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;
```
- Important assumptions and corner cases
 1- масив может быть пустым
 2- элементы могут быть отрицательными
 3- дубликаты могут быть
 4- если процесов больше чем элементов, то лишние процессы завершаються
- Memory usage considerations
 1- каждый процесс хранит свою часть масива
 2- временная память для обмена с партнёром
 3- mpi буферы для комуникации


## 6. Experimental Setup
- Процессор: ryzen 5 5600x
- Количество ядер: 6
- Количество потоков: 12
- ОЗУ: 32 Гб
- ОС: Windows 10
- Архитектура: x64

- Язык программирования: C++
- Библиотека для параллельного программирования: MPI
- Компилятор MSCV
- Тип сборки: Release 

## 7. Results and Discussion

### 7.1 Correctness
Все тесты на коретность были успешно пройдены, тестов было 9

### 7.2 Performance
 Таблица перформенс тестов:

| Mode        | processes | AvgTime(s) | Speedup | Efficiency |
|-------------|-----------|------------|---------|------------|
| seq         | 1         | 0.06400    | 1.00    | N/A        |
| mpi         | 2         | 0.02200    | 2.91    | 145%       |
| mpi         | 4         | 0.01100    | 5.82    | 145%       |
| mpi         | 6         | 0.00060    | 10.7    | 178%       |
| mpi         | 8         | 0.00040    | 16.00   | 200%       |


## 8. Conclusions
ВЫВОД :использование mpi показало свою эффективность, ускорив работу в 16 раз в максимальном варианте таким образом можно сделать вывод, что распределение нагрузки на несколько процессов является максимально эффективным метод работы с большим количеством данных
Наибольший прирост производительности наблюдается на 8 процессах
Ограничения: Не стоит делать с маленькими масивами, так как там нет выйгрыша,
Следовательно использование MPI является эффективным в условиях больших масивах когда комуникация между процессами не затратит времени больше, чем сама работа с данными

## 9. References
  MICROSOFT MPI - https://learn.microsoft.com/ru-ru/message-passing-interface/microsoft-mpi
  Parallel Programming Course - https://learning-process.github.io/parallel_programming_course/ru/index.html
  Parallel Programming 2025-2026 - https://disk.yandex.ru/d/NvHFyhOJCQU65w
  stack overflow - https://stackoverflow.com/questions
