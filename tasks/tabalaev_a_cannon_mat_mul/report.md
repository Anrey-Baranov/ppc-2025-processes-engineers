# Умножение плотных матриц. Элементы типа double. Блочная схема, алгоритм Кэннона.

- Студент: Табалаев Антон Максимович
- Технология: SEQ | MPI
- Вариант: 1

## 1. Введение 

Умножение матриц — одна из самых ресурсоемких и часто используемых операций в высокопроизводительных вычислениях. Эффективная реализация этой операции критически важна для задач моделирования физических процессов, машинного обучения, линейной алгебры и др. В данной работе рассматривается алгоритм Кэннона — блочный алгоритм умножения матриц, специально разработанный для распределенных систем с топологией «решетка».

## 2. Постановка задачи

**Цель работы:**
Реализовать последовательную и параллельную (с использованием алгоритма Кэннона) версии умножения плотных матриц, провести их сравнение и анализ эффективности.

**Определение задачи:**
Даны две квадратные матрицы `A` и `B` размера `n x n`. Необходимо вычислить результирующую матрицу `C = A × B`.

**Ограничения:**
- Матрицы должны быть квадратными одинакового размера.
- Матрицы представлены одномерными векторами типа double.
- Алгоритм Кэннона требует, чтобы количество процессов `p` было полным квадратом `(q × q = p)`, а размер матрицы `n` делился на `q` без остатка..

## 3. Алгоритм(Последовательная версия)

**Входные данные:** размер матриц `n`, вектора `a` и `b`.

**Выходные данные:** вектор `c` размера `n * n`.

**Алгоритм:**
1. Получить входные данные.
2. Инициализировать результирующий вектор нулями.
3. Использовать классический алгоритм с тройным вложенным циклом:
 - Внешние циклы i и j перебирают строки и столбцы.
 - Внутренний цикл k вычисляет скалярное произведение строки A на столбец B.
4. Вернуть вектор c.

**Сложность:** O(n^3).

### Код последовательной версии алгоритма

```
bool TabalaevACannonMatMulSEQ::RunImpl() {
  const auto n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());
  std::vector<double> c(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }

  GetOutput() = c;
  return true;
}
```

## 4. Схема распараллеливания

Алгоритм Кэннона основан на блочном разбиении матриц и их циклическом сдвиге между процессами, организованными в двумерную декартову решетку.

1. **Инициализация и проверка:**
 - Определяется размер решетки `q = sqrt(количество_процессов)`.
 - Если количество процессов не является полным квадратом или размер матрицы `n` не делится на `q` без остатка, то нулевой процесс считает результат последовательно и рассылает его через MPI_Bcast.
2. **Топология:** Для процессов создаётся двумерный декартов коммуникатор `grid_comm` с периодическими границами с помощью `MPI_Cart_create`. Каждый процесс получает свои координаты `(row col)` в решётке, которые определяют его положение и направление обмена данными.
3. **Распределение данных:** 
 - Исходные матрицы разбиваются на блоки размера `block_size x block_size`, где `block_size = n / q`.
 - Для корректного распределения подматриц используются пользовательские типы данных MPI (MPI_Type_vector и MPI_Type_create_resized):
   - `block_type` описывает один двумерный блок матрицы размером `block_size x block_size`. Он задаётся как `block_size` строк по `block_size` элементов. При этом шаг между строками равен `n` (как в исходной матрице).
   - `resized_block_type` создаётся на основе `block_type` и используется для корректировки размера типа в памяти.  Это необходимо, потому что стандартный тип, созданный с помощью `MPI_Type_vector`, включает промежутки между строками исходной матрицы. Корректировка размера, равного размеру одного элемента `double`, позволяет правильно указывать смещения `displs` и обеспечивает правильную работу распределения и сбора данных.
 - Нулевой процесс формирует вектора `sendcounts` и `displs`, а затем распределяет соответствующие блоки матриц `A` и `B` между другими процессами при помощи `MPI_Scatterv`.
4. **Начальное выравнивание блоков:** Выполняется начальный циклический сдвиг:
 - Блоки матрицы `A` сдвигаются влево на количество позиций, равное номеру строки процесса;
 - Блоки матрицы `B` сдвигаются вверх на количество позиций, равное номеру столбца процесса.
 - Сдвиги выполняются с использованием функции `MPI_Sendrecv_replace`.
5. **Основной цикл вычислений:** Алгоритм выполняется за `q` итераций. На каждой итерации:
 - Производится локальное умножение текущих блоков `A` и `B`;
 - Полученный результат накапливается в локальной матрице `C`;
 - Циклический сдвиг блоков `A` влево на 1 шаг, блоков `B` вверх на 1 шаг. Для обмена данными используется `MPI_Sendrecv_replace`.
6. **Сбор результатов:** После завершения вычислений нулевой процесс собирает локальные блоки матриц через `MPI_Gatherv`, а затем рассылает итоговую матрицу всем процессам через `MPI_Bcast`.
7. **Завершение работы:** Освобождаются пользовательские типы и декартов коммуникатор.

Код параллельной версии алгоритма представлен в приложении.

## 5. Экспериментальные исследования

### Окружение

| Параметр           | Значение                      |
|--------------------|-------------------------------|
| **OS** | Windows 11 Pro 25H2  |
| **CPU** | AMD Ryzen 5 5600X (6 cores, 12 threads, 3.70 GHz) |
| **RAM** | 16 GB DDR4 3400 МГц      |
| **Компилятор**      | MSVC 14.43.34808 |
| **Версия MPI**      | Microsoft MPI 10.1.12498.52 |

### Тестовые данные

1. Функциональные тесты:
- Используются заранее подготовленные квадратные матрицы размеров от 2x2 до 12x12.
- Элементы матриц генерируются по определённым формулам.
- Ожидаемый результат умножения вычисляется по последовательной формуле.
- Для сравнения результатов используется допустимая погрешность `epsilon = 1e-7`, что позволяет корректно проверять точность реализации.

2. Тесты производительности:
- Размер начальной матрицы: 504 × 504 (для удобства разбиения). Если тесты запускаются в MPI, то размер матрицы корректируется так, чтобы он был кратен `q = sqrt(колиество_процессов)`. Это обеспечивает равномерное распределение блоков матрицы между процессами и корректную работу алгоритма Кэннона.
- Ожидаемый результат умножения вычисляется по последовательной формуле.
- Для сравнения результатов используется допустимая погрешность `epsilon = 1e-7`, что позволяет корректно проверять точность реализации.

### Сравнение производительности

Для сравнения производительности использовалась матрица размером `1008x1008`.

Вычисления производились по следующим формулам:

- `Ускорение = Время_выполнения_на_1_процессе / Время_выполнения_на_N_процессах`
- `Эффективность = (Ускорение / Количество_процессов) × 100%`

| Режим выполнения | Количество процессов | Время выполнения (сек) | Ускорение | Эффективность |
|------------------|---------------------|------------------------|-----------|---------------|
| **MPI (1 процесс)** |||||
| pipeline | 1 | 0.2711605600 | 1.00x | 100% |
| task_run | 1 | 0.2723464400 | 1.00x | 100% |
| **MPI (4 процесса)** |||||
| pipeline | 4 | 0.0922659600 | 2.94x | 73% |
| task_run | 4 | 0.0932911400 | 2.92x | 73% |
| **MPI (9 процессов)** |||||
| pipeline | 9 | 0.0855244600 | 3.17x | 35% |
| task_run | 9 | 0.0896565800 | 3.04x | 34% |
| **MPI (16 процессов)** |||||
| pipeline | 16 | 0.2456316600 | 1.10x | 7% |
| task_run | 16 | 0.1641987600 | 1.66x | 10% |

## 6. Результаты 

1. Результаты функционального тестирования
- Все функциональные тесты успешно пройдены.
- Все реализации (SEQ и MPI) работают правильно и выдают идентичные результаты.

2. Результаты сравнения производительности
- На 4 процессах (решетка 2×2) достигнуто ускорение:
 - pipeline: 2.94× с эффективностью 73%
 - task_run: 2.92× с эффективностью 73%
- На 9 процессах (решетка 3×3) наблюдается максимальное ускорение:
 - pipeline: 3.17× при эффективности 35%
 - task_run: 3.04× при эффективности 34%
- При увеличении числа процессов до 16 (решетка 4×4) производительность резко падает:
 - pipeline: ускорение 1.10× с эффективностью 7%
 - task_run: ускорение 1.66× с эффективностью 10%
- Снижение эффективности при использовании 9 и 16 MPI-процессов обусловлено ростом накладных расходов на межпроцессорные коммуникации и аппаратными ограничениями вычислительной системы - количество запускаемых процессов превышает число физических ядер процессора.

## 7. Выводы

1. Разработанные последовательная и параллельная реализации корректно решают задачу умножения плотных матриц, что подтверждается успешным прохождением всех функциональных тестов.
2. Параллельная реализация алгоритма Кэннона с использованием MPI обеспечивает ускорение по сравнению с последовательной версией. Наиболее эффективный режим работы достигается при запуске на 4 процессах.
3. При увеличении числа MPI-процессов до 9 и 16 наблюдается снижение эффективности и ухудшение производительности, что связано с ростом накладных расходов на межпроцессные коммуникации и ограничением вычислительной системы.

## 8. Источники

1. Сысоев А. В. Лекции по курсу «Параллельное программирование для кластерных систем».
2. Официальная документация Microsoft MPI — https://learn.microsoft.com/ru-ru/message-passing-interface
3. Документация Open MPI — https://www.open-mpi.org/doc/
4. C++ Reference — https://en.cppreference.com
5. Интуит — https://intuit.ru/studies/courses/1156/190/lecture/4954?page=5
6. MPIMatr — https://edu.mmcs.sfedu.ru/file.php/74/MPIMatr.pdf
7. Нижегородский государственный университет. HPCC-ресурсы — http://www.hpcc.unn.ru/?dir=883

## 9. Приложение 

### Приложение 1 (Код параллельной версии алгоритма)

```
bool TabalaevACannonMatMulMPI::RunImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int n = 0;
  if (world_rank == 0) {
    n = static_cast<int>(std::get<0>(GetInput()));
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int q = static_cast<int>(std::sqrt(world_size));

  if (q * q != world_size || n % q != 0) {
    const size_t result_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<double> result(result_size, 0.0);
    if (world_rank == 0) {
      auto &a = std::get<1>(GetInput());
      auto &b = std::get<2>(GetInput());
      LocalMatrixMultiply(a, b, result, n);
    }
    MPI_Bcast(result.data(), static_cast<int>(result_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GetOutput() = result;

    return true;
  }

  int block_size = n / q;
  int block_elems = block_size * block_size;

  std::array<int, 2> dims = {q, q};
  std::array<int, 2> periods = {1, 1};
  MPI_Comm grid_comm = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 1, &grid_comm);

  int grid_rank = 0;
  MPI_Comm_rank(grid_comm, &grid_rank);

  std::array<int, 2> coords = {0, 0};
  MPI_Cart_coords(grid_comm, grid_rank, 2, coords.data());

  int row = coords[0];
  int col = coords[1];

  MPI_Datatype block_type = MPI_DATATYPE_NULL;
  MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &block_type);
  MPI_Type_commit(&block_type);
  MPI_Datatype resized_block = MPI_DATATYPE_NULL;
  MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block);
  MPI_Type_commit(&resized_block);

  std::vector<double> local_a(block_elems);
  std::vector<double> local_b(block_elems);
  std::vector<double> local_c(block_elems, 0.0);

  std::vector<int> counts(world_size, 1);
  std::vector<int> displs(world_size);

  if (grid_rank == 0) {
    for (int i = 0; i < q; ++i) {
      for (int j = 0; j < q; ++j) {
        displs[(i * q) + j] = (i * n * block_size) + (j * block_size);
      }
    }
    auto &a = std::get<1>(GetInput());
    auto &b = std::get<2>(GetInput());
    MPI_Scatterv(a.data(), counts.data(), displs.data(), resized_block, local_a.data(), block_elems, MPI_DOUBLE, 0,
                 grid_comm);
    MPI_Scatterv(b.data(), counts.data(), displs.data(), resized_block, local_b.data(), block_elems, MPI_DOUBLE, 0,
                 grid_comm);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, resized_block, local_a.data(), block_elems, MPI_DOUBLE, 0, grid_comm);
    MPI_Scatterv(nullptr, nullptr, nullptr, resized_block, local_b.data(), block_elems, MPI_DOUBLE, 0, grid_comm);
  }

  int left = 0;
  int right = 0;
  int up = 0;
  int down = 0;

  MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
  MPI_Cart_shift(grid_comm, 0, 1, &up, &down);

  for (int i = 0; i < row; ++i) {
    MPI_Sendrecv_replace(local_a.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
  }
  for (int i = 0; i < col; ++i) {
    MPI_Sendrecv_replace(local_b.data(), block_elems, MPI_DOUBLE, up, 1, down, 1, grid_comm, MPI_STATUS_IGNORE);
  }

  for (int k = 0; k < q; ++k) {
    LocalMatrixMultiply(local_a, local_b, local_c, block_size);
    MPI_Sendrecv_replace(local_a.data(), block_elems, MPI_DOUBLE, left, 0, right, 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(local_b.data(), block_elems, MPI_DOUBLE, up, 1, down, 1, grid_comm, MPI_STATUS_IGNORE);
  }

  std::vector<double> global_result(static_cast<size_t>(n) * static_cast<size_t>(n));
  MPI_Gatherv(local_c.data(), block_elems, MPI_DOUBLE, global_result.data(), counts.data(), displs.data(),
              resized_block, 0, grid_comm);

  MPI_Bcast(global_result.data(), n * n, MPI_DOUBLE, 0, grid_comm);
  GetOutput() = global_result;

  MPI_Type_free(&resized_block);
  MPI_Type_free(&block_type);
  MPI_Comm_free(&grid_comm);
  return true;
}

void TabalaevACannonMatMulMPI::LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b,
                                                   std::vector<double> &c, int n) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double temp = a[(i * n) + k];
      for (int j = 0; j < n; ++j) {
        c[(i * n) + j] += temp * b[(k * n) + j];
      }
    }
  }
}
```