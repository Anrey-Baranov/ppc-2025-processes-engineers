# Решение систем линейных уравнений методом сопряженных градиентов

-   Студент: Самойленко Илья Андреевич, группа 3823Б1ПР3
-   Технология: SEQ|MPI
-   Вариант: 6

## 1. Введение

Задача состоит в реализации последовательной и параллельной версий метода сопряжённых градиентов для решения системы линейных алгебраических уравнений, а также создание функциональных тестов для проверки корректности решения и тестов на производительность для сравнения последовательной и параллельной реализаций метода.

## 2. Постановка задачи

Дана система линейных уравнений Ax = b.
Матрица системы A - это действительная, симметричная и положительно определённая матрица.
Необходимо найти вектор x, удовлетворяющий уравнению с заданной точностью $\epsilon$.

В этой реализации матрица A (матрица коэффициентов) имеет 3 различных варианта генерации.

Вариант 1 (Тридиагональная):
-   Элементы главной диагонали равны 4.0.
-   Элементы дополнительных диагоналей равны 1.0.
-   Остальные элементы равны 0.0.

Вариант 2 (Диагональная):
-   Элементы главной диагонали равны 5.0.
-   Остальные элементы равны 0.0.

Вариант 1 (Центросимметричная):
-   Элементы главной диагонали равны 3.0.
-   Элементы побочной диагонали равны -1.0.
-   Остальные элементы равны 0.0.

Вектор b (вектор свободных членов) инициализируется значениями 1.0.

Метод сопряжённых градиентов - это алгоритм для численного решения систем линейных уравнений и основан на минимизации квадратичной функции:

$$f(x) = \frac{1}{2} x^T A x - b^T x$$

Градиент этой функции: $\nabla f(x) = Ax - b = -r$ ($r$ - вектор невязки). Метод заключается в итеративном вычислении нового приближения вектора x по формулам:

$$\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$$
$$x_{k+1} = x_k + \alpha_k p_k$$
$$r_{k+1} = r_k - \alpha_k A p_k$$
$$\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$$
$$p_{k+1} = r_{k+1} + \beta_k p_k$$

где $r$ - вектор невязки, $p$ - направление поиска.

Выход из цикла итераций происходит при выполнении условия $\lVert r_k \rVert_2 < \epsilon$ или достижении максимального количества итераций.

В данной реализации:

-   $\epsilon = 10^{-7}$
-   Максимальное количество итераций = 2000
-   Начальное приближение $x^{(0)}$ - нулевой вектор.

Входные данные: пара (размер матрицы N, вариант матрицы V).

Выходные данные: вектор решения x.

## 3. Базовый алгоритм (Последовательный)

Инициализация:

-   Выбрать начальное приближение $x_0$
-   Вычислить начальную невязку: $r_0 = b - Ax_0$
-   Установить начальное направление поиска: $p_0 = r_0$

Итерация:

1. Вычислить размер шага:
   $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$

2. Обновить решение:
   $x_{k+1} = x_k + \alpha_k p_k$

3. Обновить невязку:
   $r_{k+1} = r_k - \alpha_k A p_k$

4. Вычислить коэффициент сопряжённости:
   $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$

5. Обновить направление поиска:
   $p_{k+1} = r_{k+1} + \beta_k p_k$

```cpp
for (int it = 0; it < iters; ++it) {
  if (std::sqrt(res_dot) < eps) {
    break;
  }

  MatrixVectorMult(size, matrix, dir, matdir);

  double matdir_dot = DotProduct(dir, matdir);
  double step = res_dot / matdir_dot;

  for (size_t i = 0; i < size; ++i) {
    x[i] += step * dir[i];
    res[i] -= step * matdir[i];
  }

  double res_dot_new = DotProduct(res, res);
  double conj_coef = res_dot_new / res_dot;

  for (size_t i = 0; i < size; ++i) {
    dir[i] = res[i] + conj_coef * dir[i];
  }

  res_dot = res_dot_new;
}
```

## 4. Схема распараллеливания

Параллельная версия использует декомпозицию матрицы по строкам.

-   Матрица A распределяется по строкам между процессами.
-   Вектор b (вектор свободных членов) распределяется между процессами, где каждый процесс получает свою локальную часть.
-   Основные векторы метода (x, res, dir) хранятся и обновляются локально.
-   Для умножения матрицы на вектор ($Ap$) требуется полный вектор направления $p$, поэтому перед умножением выполняется составление полного вектора.
-   Скалярные произведения вычисляются суммой локальных скалярных произведений.

```cpp
MPI_Allgatherv(local_dir.data(), local_rows, MPI_DOUBLE, dir.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
LocalMatrixVectorMult(size, local_rows, local_matrix, dir, local_matdir);

double local_dir_dot = LocalDotProduct(local_rows, local_dir, local_matdir);
double dir_dot = 0.0;
MPI_Allreduce(&local_dir_dot, &dir_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

## 5. Детали реализации

Структура кода

```
tasks/samoylenko_i_conj_grad_method/
├───common
│   └───include
│           common.hpp // Определение типов данных
├───mpi
│   ├───include
│   │       ops_mpi.hpp
│   └───src
│           ops_mpi.cpp // Параллельная реализация
├───seq
│   ├───include
│   │       ops_seq.hpp
│   └───src
│           ops_seq.cpp // Последовательная реализация
└───tests
    ├───functional
    │       main.cpp // Функциональные тесты
    └───performance
            main.cpp // Тесты на производительность
```

Краевые случаи:

-   Нулевой или отрицательный размер системы (N <= 0).
-   Количество процессов больше размера системы (world_size > N)
-   Минимальный размер системы (N = 1)
-   Несуществующий вариант матрицы (V > 2 или V < 0)

Особенности использования памяти:

-   В параллельной версии каждый процесс хранит только свои строки матрицы A, локальные части векторов x, res, dir, matdir и полные вектора x, dir (на моменты коммуникации).

## 6. Экспериментальная настройка

Аппаратное обеспечение и ОС:

-   CPU: AMD Ryzen 5 2600X
-   ядра: 6
-   потоки: 12
-   RAM: 16 GB
-   OS: Ubuntu 24.04.2 LTS (DevContainer / Windows 10 22H2)

Набор инструментов:

-   Компилятор: g++ 13.3.0
-   Тип сборки: Release
-   MPI: OpenMPI 4.1.6

Переменные окружения:

-   OMPI_ALLOW_RUN_AS_ROOT=1
-   OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

Данные: Для тестов на производительность размер системы: N = 3500.

## 7. Результаты и обсуждение

### 7.1 Корректность

Функциональные тесты проверяют корректность вычисления решения путем подстановки найденного вектора x в исходную систему уравнений и вычисления невязки $r = Ax - b$. Критерием корректности является выполнение условия $\lVert r \rVert_\infty < \epsilon$. Все тесты проходят успешно. Отдельно тестируются краевые случаи и различные варианты генерации матриц.

### 7.2 Производительность

| Режим | Процессы | Время, с | Ускорение | Эффективность |
| ----- | -------- | -------- | --------- | ------------- |
| seq   | 1        | 1.12     | 1.00      | N/A           |
| mpi   | 1        | 1.12     | 1.00      | 100%          |
| mpi   | 2        | 0.57     | 1.96      | 98%           |
| mpi   | 4        | 0.29     | 3.86      | 96.5%         |

Ускорение = Время seq / Время mpi

Эффективность = Ускорение / Процессы \* 100%

## 8. Заключение

Реализован метод сопряжённых градиентов для решения СЛАУ. MPI-версия использует декомпозицию данных по строкам и распределение векторов. Обе реализации прошли все тесты на функционал, включая поддержку различных типов матриц. MPI-версия масштабируется корректно.

## 9. Источники

1. Conjugate gradient method: https://en.wikipedia.org/wiki/Conjugate_gradient_method
2. Метод сопряжённых градиентов: https://ru.wikipedia.org/wiki/Метод_сопряжённых_градиентов_(СЛАУ)
3. Материалы курса: https://learning-process.github.io/parallel_programming_course/ru/index.html
4. Документация Open MPI: https://www-lb.open-mpi.org/doc/

## Приложение

```cpp
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int n = 0;
  int variant = 0;
  if (world_rank == 0) {
    n = GetInput().first;
    variant = GetInput().second;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&variant, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return false;
  }

  auto size = static_cast<size_t>(n);

  std::vector<int> row_counts(world_size);
  std::vector<int> row_displs(world_size);
  CalculateDistribution(size, world_size, row_counts, row_displs);

  int local_rows = row_counts[world_rank];
  int local_start = row_displs[world_rank];

  std::vector<double> local_matrix = BuildLocalMatrix(size, local_rows, local_start, variant);
  std::vector<double> vector(size);
  if (world_rank == 0) {
    for (size_t i = 0; i < size; ++i) {
      vector[i] = 1.0;
    }
  }

  std::vector<double> local_vector(local_rows);
  MPI_Scatterv(vector.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, local_vector.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_x(local_rows, 0.0);

  ConjugateGradient(size, local_rows, local_matrix, local_vector, row_counts, row_displs, local_x);

  std::vector<double> x(size);
  MPI_Gatherv(local_x.data(), local_rows, MPI_DOUBLE, x.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = x;
  }

  return true;
```
