# Умножение разреженных матриц. Элементы типа double. Формат хранения матрицы – столбцовый (CCS).

- Студент: Баданов Александр, группа 3823Б1ПР2
- Технология: SEQ, MPI
- Вариант: 5

## 1. Введение
Умножение разреженных матриц - одна из ключевых операций в вычислительной математике, особенно востребованная при решении систем линейных уравнений, обработке графов и машинном обучении. Формат хранения CCS (Compressed Column Storage) позволяет эффективно хранить и обрабатывать разреженные матрицы за счёт сжатия ненулевых элементов по столбцам. Данная работа реализует параллельное умножение разреженных матриц в формате CCS с использованием MPI для распределения вычислений между узлами.

## 2. Постановка задачи
Разработать последовательную (SEQ) и параллельную (MPI) реализации умножения двух разреженных матриц, хранящихся в формате CCS, с элементами типа double.

Входные данные:
- Значения, индексы строк и указатели столбцов матрицы A
- Значения, индексы строк и указатели столбцов матрицы B
- Размеры матриц: rows_a, cols_a (совпадает с rows_b), cols_b

Выходные данные:
- Значения, индексы строк и указатели столбцов результирующей матрицы C

## 3. Базовый алгоритм (Последовательный)
```cpp
bool BadanovASparseMatrixMultDoubleCcsSEQ::RunImpl() {
  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &value_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  SparseMatrix a;
  a.values = values_a;
  a.row_indices = row_indices_a;
  a.col_pointers = col_pointers_a;
  a.rows = rows_a;
  a.cols = cols_a;

  SparseMatrix b;
  b.values = value_b;
  b.row_indices = row_indices_b;
  b.col_pointers = col_pointers_b;
  b.rows = cols_a;
  b.cols = cols_b;

  SparseMatrix c = MultiplyCCS(a, b);

  GetOutput() = std::make_tuple(c.values, c.row_indices, c.col_pointers);

  return true;
}
```
Последовательная версия умножает две разреженные матрицы в формате CCS, предварительно развернув столбцы матрицы A в плотный вид для ускорения доступа, и вычисляет результирующую матрицу C, отсекая значения меньше 1e-10.

## 4. Схема распараллеливания
### Распределение данных
Используется горизонтальное распределение строк матрицы A между процессами MPI:
- Матрица A делится по строкам на примерно равные блоки
- Каждый процесс получает свой блок строк A и полную копию матрицы B
- Результат (блок строк C) собирается на всех процессах

### Коммуникационная схема

1. Распределение данных:
- Корневой процесс (ранг 0) рассылает блоки A и полную B (в данной реализации предполагается, что B доступна всем изначально)
2. Локальное умножение:
- Каждый процесс умножает свой блок строк A на матрицу B, получая блок строк C
3. Сбор результатов:
- Используется MPI_Gatherv для сбора значений и индексов по столбцам
- MPI_Allreduce для подсчёта общего числа ненулевых элементов в столбцах

### Алгоритм локального умножения
```cpp
std::vector<double> SparseDotProduct(const SparseMatrix &a, const SparseMatrix &b, int col_b) {
  std::vector<double> result(a.rows, 0.0);
  for (int idx_b = b.col_pointers[col_b]; idx_b < b.col_pointers[col_b + 1]; ++idx_b) {
    int row_b = b.row_indices[idx_b];
    double val_b = b.values[idx_b];
    for (int idx_a = a.col_pointers[row_b]; idx_a < a.col_pointers[row_b + 1]; ++idx_a) {
      int row_a = a.row_indices[idx_a];
      double val_a = a.values[idx_a];
      result[row_a] += val_a * val_b;
    }
  }
  return result;
}
```

## 5. Детали реализации
### Структура кода
- `common.hpp` - общие типы данных
- `ops_mpi.cpp` - MPI реализация
- `ops_seq.cpp` - SEQ реализация
- Тесты в папках `tests/functional/` и `tests/performance/`

### Особенности реализации
- Поддержка неравномерного распределения строк при неделящемся количестве процессов
- Пороговая фильтрация малых значений (1e-10)
- Эффективный сбор данных с использованием MPI_Gatherv и смещений (displs)

## 6. Экспериментальная установка
### Оборудование и ПО
- **Процессор:** Apple M1
- **ОС:** macOS 15.3.1
- **Компилятор:** clang version 21.1.5
- **Тип сборки:** release
- **MPI:** Open MPI v5.0.8

### Данные для тестирования
Функциональные тесты (15 случаев):
- Размеры матриц от 10×10×10 до 800×800×800
- Различные соотношения размеров: квадратные, прямоугольные
- Плотность заполнения ~5%

Производительные тесты:
- Small: 50×50×50
- Medium: 200×200×200
- Large: 500×500×500
- Huge: 1000×1000×1000

## 7. Результаты и обсуждение
### 7.1 Проверка корректности
Все 15 функциональных тестов пройдены:
- Корректность размеров выходных данных
- Соответствие индексов допустимым диапазонам
- Согласованность формата CCS (размеры массивов, монотонность указателей)

### 7.2 Производительность
Производительность оценивалась с помощью четырёх тестовых конфигураций:
- Small (500×500×500) — базовый случай для оценки накладных расходов MPI
- Medium (2000×2000×2000) — типичный рабочий размер для оценки эффективности
- Large (5000×5000×5000) — тест на масштабируемость при высокой нагрузке
- Huge (10000×10000×10000) — проверка ограничений по памяти и устойчивости

| Процессы | Время, с | Ускорение | Эффективность |
|----------|-----------|-----------|---------------|
| 1 (SEQ)  |    0,36   | 1.00      | N/A           |
| 2        |    0,40   | 0,90      | 45%           |
| 4        |    0,54   | 0,67      | 17%           |
| 8        |    0,79   | 0,46      | 6%            |


## 8. Выводы
В ходе работы была успешно решена задача умножения разреженных матриц с использованием последовательного алгоритма и технологии MPI для параллельных вычислений
- Реализованы корректные SEQ и MPI-версии умножения разреженных матриц в формате CCS.
- MPI-реализация демонстрирует принцип горизонтального распределения строк с сохранением локальности данных.
- Алгоритм эффективен для матриц с умеренной разреженностью (1–10% ненулевых элементов).

### Ограничения
- Накладные расходы MPI
- Снижение эффективности при большом количестве процессов
- Хранение полной матрицы B на каждом процессе, что ограничивает масштабируемость по памяти


## 9. Источники
1. Курс лекций по параллельному программированию Сысоева Александра Владимировича. 
2. Документация по курсу: https://learning-process.github.io/parallel_programming_course/ru

## Приложение

```cpp
bool BadanovASparseMatrixMultDoubleCcsMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();

  const auto &values_a = std::get<0>(in);
  const auto &row_indices_a = std::get<1>(in);
  const auto &col_pointers_a = std::get<2>(in);
  const auto &values_b = std::get<3>(in);
  const auto &row_indices_b = std::get<4>(in);
  const auto &col_pointers_b = std::get<5>(in);
  int rows_a = std::get<6>(in);
  int cols_a = std::get<7>(in);
  int cols_b = std::get<8>(in);

  SparseMatrix a_global;
  a_global.values = values_a;
  a_global.row_indices = row_indices_a;
  a_global.col_pointers = col_pointers_a;
  a_global.rows = rows_a;
  a_global.cols = cols_a;

  SparseMatrix b_global;
  b_global.values = values_b;
  b_global.row_indices = row_indices_b;
  b_global.col_pointers = col_pointers_b;
  b_global.rows = cols_a;
  b_global.cols = cols_b;

  LocalData local_data = DistributeDataHorizontal(world_rank, world_size, a_global, b_global);

  SparseMatrix local_c = MultiplyLocal(local_data);

  SparseMatrix global_c;
  GatherResults(world_rank, world_size, local_c, global_c);

  GetOutput() = std::make_tuple(global_c.values, global_c.row_indices, global_c.col_pointers);

  return true;
}
```