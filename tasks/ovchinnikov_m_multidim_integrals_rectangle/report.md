# Вычисление многомерных интегралов методом прямоугольников

- Student: Овчинников Матвей Евгеньевич, group 3823Б1ПР2
- Technology: SEQ | MPI
- Variant: 7

## 1. Introduction

- **Brief motivation:** Изучить методы численного интегрирования в многомерных пространствах, освоить технику параллелизации сложных по вычислению задач с использованием MPI, реализовать эффективный алгоритм вычисления многомерных интегралов.
- **Problem context:** Сравнить производительность последовательной и параллельной реализаций метода прямоугольников для вычисления многомерных интегралов, исследовать влияние количества измерений и разбиений на точность и скорость вычислений.
- **Expected outcome:** Получить практический опыт работы с многомерными вычислениями, изучить методы распределения нагрузки между процессами при решении задач численного интегрирования.

## 2. Problem Statement

- **Formal task definition:** Реализовать последовательную и параллельную версии программы для вычисления многомерного интеграла функции суммы квадратов на гиперпрямоугольной области методом прямоугольников. Оценить производительность и точность обеих реализаций.
- **input/output format:** На вход подаётся количество разбиений (n), размерность (dim), вектор нижних границ, вектор верхних границ. На выходе мы получаем вещественное число, которое является значением интеграла.

## 3. Baseline Algorithm (Sequential)

Базовый алгоритм использует метод прямоугольников для приближенного вычисления многомерного интеграла. Для каждой размерности создается сетка из n точек (центры прямоугольников). Алгоритм перебирает все комбинации точек по разным измерениям, вычисляет значение функции в каждой точке и суммирует вклад каждого прямоугольника.

```cpp
bool OvchinnikovMMultiDimIntegralsRectangleSEQ::RunImpl() {
  const auto &input = GetInput();
  int n = std::get<0>(input);
  int dim = std::get<1>(input);
  const auto &lower_bounds = std::get<2>(input);
  const auto &upper_bounds = std::get<3>(input);

  std::vector<std::vector<double>> axis_coords(dim);
  double cell_volume = 1.0;

  for (int curr_dim = 0; curr_dim < dim; curr_dim++) {
    double step = (upper_bounds[curr_dim] - lower_bounds[curr_dim]) / n;
    cell_volume *= step;

    axis_coords[curr_dim].resize(n);
    for (int i = 0; i < n; i++) {
      axis_coords[curr_dim][i] = lower_bounds[curr_dim] + ((i + 0.5) * step);
    }
  }

  double integral = 0.0;
  std::vector<double> point(dim);
  std::vector<int> indices(dim, 0);

  bool done = false;
  while (!done) {
    for (int curr_dim = 0; curr_dim < dim; curr_dim++) {
      point[curr_dim] = axis_coords[curr_dim][indices[curr_dim]];
    }

    integral += MultivariableFunction(point);

    int d = dim - 1;
    while (d >= 0) {
      indices[d]++;
      if (indices[d] < n) {
        break;
      }
      indices[d] = 0;
      d--;
    }

    if (d < 0) {
      done = true;
    }
  }

  integral *= cell_volume;
  GetOutput() = integral;
  return true;
}
```

Алгоритм использует эффективный способ обхода всех комбинаций индексов с помощью вложенного цикла. Это позволяет отказаться от использования рекурсии, которая будет неэффективной при больших размерностях.

## 4. Parallelization Scheme

- **Data distribution:**
Данные распределяются нулевым процессом между остальными процессами следующим образом:
1. Вычисляется общее количество точек интегрирования: total_points = n^dim
2. Определяется базовое количество точек на процесс: points_per_process = total_points / size
3. Вычисляется остаток: remainder = total_points % size
4. Каждый процесс получает диапазон точек для обработки. Процессы с номерами меньше remainder получают на одну точку больше, это реализовано через переменную extra_points:
```cpp
 int extra_points = static_cast<int>(remainder > 0);
```

## **Rank roles:**

### **Process 0:**

#### 1. Читает входные данные и вычисляет основные параметры (шаги, объем ячейки)

#### 2. Рассылает параметры всем процессам через MPI_Bcast

```cpp
std::array<int, 2> params = {n, dim};
MPI_Bcast(params.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

std::vector<double> all_bounds;
all_bounds.insert(all_bounds.end(), lower_bounds.begin(), lower_bounds.end());
all_bounds.insert(all_bounds.end(), upper_bounds.begin(), upper_bounds.end());
MPI_Bcast(all_bounds.data(), 2 * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

MPI_Bcast(&cell_volume, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

#### 3. Вычисляет распределение точек по процессам (Описано в data distribution)

#### 4. Отправляет каждому процессу его диапазон через MPI_Send

```cpp
for (int i = 1; i < size; i++) {
    start_point = 0;
    for (int j = 0; j < i; j++) {
        int extra_points_for_j = static_cast<int>(j < remainder);
        int points_for_j = points_per_process + extra_points_for_j;
        start_point += points_for_j;
    }
    extra_points = static_cast<int>(i < remainder);
    end_point = start_point + points_per_process + extra_points;

    std::array<int, 2> range = {start_point, end_point};
    MPI_Send(range.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
}
```

#### 5. Собирает результаты от всех процессов через MPI_Recv и суммирует

```cpp
global_integral = local_integral;

for (int i = 1; i < size; i++) {
    double partial_result = 0.0;
    MPI_Recv(&partial_result, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    global_integral += partial_result;
}

GetOutput() = global_integral;
```

#### 6. Рассылает итоговый результат через MPI_Bcast

```cpp
MPI_Bcast(&global_integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

### **Other processes:**

#### 1. Получают параметры от процесса 0 через MPI_Bcast

```cpp
std::array<int, 2> params{{0, 0}};
MPI_Bcast(params.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
n = params[0];
dim = params[1];

std::vector<double> all_bounds(static_cast<size_t>(2) * dim);
MPI_Bcast(all_bounds.data(), 2 * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

std::vector<double> local_lower_bounds(all_bounds.begin(), all_bounds.begin() + dim);
std::vector<double> local_upper_bounds(all_bounds.begin() + dim, all_bounds.end());

MPI_Bcast(&cell_volume, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

#### 2. Получают свой диапазон точек через MPI_Recv

```cpp
std::array<int, 2> range{{0, 0}};
MPI_Recv(range.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

int start_point = range[0];
int end_point = range[1];
```

#### 3. Вычисляют свою часть интеграла

```cpp
std::vector<double> local_steps(dim);
for (int i = 0; i < dim; i++) {
    local_steps[i] = (local_upper_bounds[i] - local_lower_bounds[i]) / n;
}
local_integral = ComputePartialIntegral(func, n, dim, local_lower_bounds, 
                                       local_steps, cell_volume, start_point, end_point);
```

#### 4. Отправляют результат процессу 0 через MPI_Send

```cpp
MPI_Send(&local_integral, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
```

## **Особенности вычисления частичного интеграла:**

```cpp
double ComputePartialIntegral(const std::function<double(const std::vector<double> &)> &func, 
                              int n, int dim,
                              const std::vector<double> &lower_bounds, 
                              const std::vector<double> &steps,
                              double cell_volume, 
                              int start_point, int end_point) {
    // Преобразование линейного индекса в многомерные координаты
    for (int idx = start_point; idx < end_point; idx++) {
        int temp = idx;
        for (int i = dim - 1; i >= 0; i--) {
            indices[i] = temp % n;  // получаем индекс по измерению i
            temp /= n;              // переходим к следующему измерению
        }
        // Вычисление координат точки
        for (int i = 0; i < dim; i++) {
            point[i] = lower_bounds[i] + ((indices[i] + 0.5) * steps[i]);
        }
        // Добавление вклада точки в интеграл
        partial_integral += func(point) * cell_volume;
    }
    return partial_integral;
}
```
Этот подход позволяет каждому процессу эффективно вычислять свой диапазон точек без необходимости хранить всю многомерную сетку в памяти.

## 5. Implementation Details

- **Code structure (files, key classes/functions):**
    - `DefaultFunction()` - интегрируемая функция (я выбрал сумму квадратов, а по скольку это вынесено в отдельную функцию, то её легко поменять)
    ```cpp
    double DefaultFunction(const std::vector<double> &point) {
      double sum = 0.0;
      for (double coord : point) {
        sum += coord * coord;
      }
      return sum;
    }
    ```
    - `ComputePartialIntegral()` - вычисление локального интеграла в заданном диапазоне точек
    ```cpp
    double ComputePartialIntegral(const std::function<double(const std::vector<double> &)> &func, int n, int dim,
                              const std::vector<double> &lower_bounds, const std::vector<double> &steps,
                              double cell_volume, int start_point, int end_point) {
      double partial_integral = 0.0;

      if (start_point >= end_point) {
        return partial_integral;
      }
      std::vector<int> indices(dim);
      std::vector<double> point(dim);
      for (int idx = start_point; idx < end_point; idx++) {
        int temp = idx;
        for (int i = dim - 1; i >= 0; i--) {
          indices[i] = temp % n;
          temp /= n;
        }

        for (int i = 0; i < dim; i++) {
          point[i] = lower_bounds[i] + ((indices[i] + 0.5) * steps[i]);
        }
        partial_integral += func(point) * cell_volume;
      }

      return partial_integral;  
    }
    ```
    - `RunImpl()` - основная логика (в пункте 4. Parallelization Scheme подробно расписана её реализация с примерами кода)
    - `ValidationImpl()` - проверка входных данных на валидность
    ```cpp
    bool OvchinnikovMMultiDimIntegralsRectangleMPI::ValidationImpl() {
      const auto &input = GetInput();
      if (std::get<0>(input) <= 0) {
        return false; 
      }
      if (std::get<1>(input) <= 0) {
        return false; 
      }

      const auto &lower_bounds = std::get<2>(input);
      const auto &upper_bounds = std::get<3>(input);
      int dim = std::get<1>(input);
      if (lower_bounds.size() != static_cast<size_t>(dim)) {
        return false;
      }
      if (upper_bounds.size() != static_cast<size_t>(dim)) {
        return false;
      }

      for (int i = 0; i < dim; i++) {
        if (lower_bounds[i] >= upper_bounds[i]) {
          return false;
        }
      }
      return true;
    }
    ```

- **Important assumptions and corner cases:**
  - Количество разбиений n должно быть положительным
  - Размерность dim должна быть положительной
  - Векторы границ должны иметь размер, равный dim
  - Нижние границы должны быть строго меньше верхних
  - При невалидных данных функции ValidationImpl возвращают false

- **Memory usage considerations:**
  - SEQ версия хранит все координатные сетки в памяти
  - MPI версия распределяет память между процессами
  - Каждый процесс хранит только свои данные и необходимые параметры

## 6. Experimental Setup

- **Hardware/OS:**
  - Модель ЦП: 13th Gen Intel(R) Core(TM) i7-13700H
  - Архитектура: x86_64
  - Ядра/потоки: 6 ядер, 12 потоков
  - ОЗУ: 16 ГБ
  - Версия ОС: WSL: 2.6.1.0
  - Ядро: Linux 6.8.0-87-generic

- **Toolchain:**
  - GCC 13.1.0 (Ubuntu 13.1.0-8ubuntu1~22.04)
  - Clang 18.1.3
  - CMake 3.28.3
  - GNU Make 4.3
  - Open MPI 4.1.5

## 7. Results and Discussion

### 7.1 Correctness

Корректность работы была проверена с помощью комплексного функционального тестирования, включающего 27 тестовых случаев на SEQ и MPI версии программы:

```cpp
const std::array<TestType, 27> kTestParam = {
    std::make_tuple(-10, 1, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(10, -2, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(-5, -2, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(0, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(10, 1, std::vector<double>{0.0}, std::vector<double>{1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(20, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{2.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{-1.0, 0.0}, std::vector<double>{1.0, 2.0}),
    std::make_tuple(20, 3, std::vector<double>{0.0, 1.0, -1.0}, std::vector<double>{2.0, 3.0, 1.0}),
    std::make_tuple(5, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(10, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(20, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(40, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(8, 4, std::vector<double>{0.0, 0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0}),
    std::make_tuple(6, 5, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}),
    std::make_tuple(5, 4, std::vector<double>{0.0, 0.0, 0.0, 0.0}, std::vector<double>{2.0, 2.0, 2.0, 2.0}),
    std::make_tuple(2, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(3, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(4, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(10, 2, std::vector<double>{-2.0, 0.5}, std::vector<double>{2.0, 3.5}),
    std::make_tuple(12, 3, std::vector<double>{-1.0, 0.0, 1.0}, std::vector<double>{1.0, 2.0, 3.0}),
    std::make_tuple(30, 2, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}),
    std::make_tuple(25, 3, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}),
    std::make_tuple(15, 2, std::vector<double>{0.5, 1.0}, std::vector<double>{2.5, 3.0}),
    std::make_tuple(18, 3, std::vector<double>{0.1, 0.2, 0.3}, std::vector<double>{1.1, 1.2, 1.3})};
```
Тесты проверяют разные значения размерности, кол-ва разбиений и границ. Входные данные проверяются на валидность, а итоговый результат сравнивается с учетом возможной погрешности (метод прямоугольников не идеален и чем меньше кол-во разбиений, тем больше погрешность. Например, при одном прямоугольнике погрешность может достигать 30%, поэтому в функциональных тестах нет случаев с одним прямоугольником).

Возможная погрешность рассчитывается с помощью функции `CalculateError`, точность решения зависит от n и dim напрямую:

```cpp
double CalculateError(int n, int dim, const std::vector<double> &lower_bounds,
                      const std::vector<double> &upper_bounds) {
  double base_error = 1.0 / (n * n);
  double dim_factor = std::sqrt(static_cast<double>(dim));
  double max_range = 0.0;
  for (int i = 0; i < dim; i++) {
    max_range = std::max(max_range, upper_bounds[i] - lower_bounds[i]);
  }
  double final_error = base_error * dim_factor * max_range;
  return std::max(1e-8, std::min(final_error, 0.1));
}
```

### 7.2 Performance

Производительность оценивалась для 3 измерений с количеством разбиений 300. Результаты показывают значительное ускорение при использовании MPI:

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|------------------|---------|------------|
| seq  | 1         | 0.2058804989 | 1.00 | N/A |
| seq  | 1         | 0.2108461857 | 1.00 | N/A |
| mpi  | 1         | 0.2581463916 | 0.81 | 80.6% |
| mpi  | 1         | 0.2589943230 | 0.80 | 80.4% |
| mpi  | 2         | 0.2614476926 | 0.80 | 39.9% |
| mpi  | 2         | 0.2592154988 | 0.80 | 40.2% |
| mpi  | 3         | 0.1743425244 | 1.20 | 39.8% |
| mpi  | 3         | 0.1752907180 | 1.19 | 39.6% |
| mpi  | 4         | 0.1361768702 | 1.53 | 38.3% |
| mpi  | 4         | 0.1312491762 | 1.59 | 39.7% |
| mpi  | 6         | 0.0959415378 | 2.17 | 36.2% |
| mpi  | 6         | 0.0930659908 | 2.24 | 37.3% |
| mpi  | 8         | 0.0879832104 | 2.37 | 29.6% |
| mpi  | 8         | 0.0837573880 | 2.49 | 31.1% |
| mpi  | 10        | 0.0718120576 | 2.90 | 29.0% |
| mpi  | 10        | 0.0746595884 | 2.79 | 27.9% |

Мы наблюдаем ускорение работы программы уже при использовании трёх процессов. В пике достигается уменьшение времени работы почти в 3 раза, однако эффективность использования такого кол-ва процессов очень низкая.


## 8. Conclusions

В ходе работы успешно реализованы SEQ и MPI версии алгоритма вычисления многомерных интегралов методом прямоугольников. Параллельная версия демонстрирует значительное ускорение при сохранении приемлемой точности вычислений.

**Основные достижения:**
1. Реализован эффективный алгоритм обхода многомерной сетки без использования рекурсии
2. Разработана схема равномерного распределения точек интегрирования между процессами
3. Обеспечена корректная обработка граничных случаев и невалидных данных

**Ограничения и проблемы:**
1. Метод прямоугольников имеет низкий порядок точности O(h²)
2. Кол-во точек растет экспоненциально с ростом размерности
3. При очень большом кол-ве точек возможны переполнения при вычислении total_points (во избежания этого используется int64_t)
4. Из-за большого кол-ва рассылок данных использование параллельной версии алгоритма неэффективно 


## 9. Ссылки

1. Документация Open MPI
2. Учебные видеоматериалы по параллельному программированию
3. Численные методы: метод прямоугольников для многомерного интегрирования
4. Записи онлайн лекций Сысоева Александра Владимировича