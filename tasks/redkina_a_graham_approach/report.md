# "Построение выпуклой оболочки — проход Грэхема", вариант № 24
### Студент: Редькина Алина Александровна
### Группа: 3823Б1ПР1
### Преподаватель: Сысоев Александр Владимирович, доцент


## Введение

  Вычисление выпуклой оболочки множества точек является одной из фундаментальных задач вычислительной геометрии. Выпуклая оболочка представляет собой минимальный выпуклый многоугольник, содержащий все заданные точки на плоскости.Алгоритм Грэхема является классическим и широко используемым методом построения выпуклой оболочки. 

---

## Постановка задачи

**Цель работы:**  
  Реализовать алгоритм Грэхема для построения выпуклой оболочки множества точек на плоскости в последовательном и параллельном (MPI) вариантах.

**Определение задачи:**  
  Дано множество точек `P = {p1, p2, ..., pn}` на плоскости. Требуется построить выпуклую оболочку — упорядоченный набор точек, образующих выпуклый многоугольник, содержащий все точки множества `P`.

**Ограничения:**
  - Каждая точка задаётся целочисленными координатами `(x, y)`;
  - Минимальное число точек для построения оболочки — 3;
  - Поддерживается корректная обработка вырожденных случаев (коллинеарные точки);
  - Результат MPI-версии должен совпадать с результатом последовательной реализации.

---

## Описание алгоритма (последовательная версия)
  Последовательная реализация основана на классическом алгоритме Грэхема.

Алгоритм:
  1. Поиск опорной точки: выбирается точка с минимальной координатой `y` (при равенстве — с минимальной `x`).
  2. Сортировка точек: остальные точки сортируются по возрастанию полярного угла относительно опорной точки. В случае равенства углов приоритет отдаётся ближайшей точке.
  3. Построение оболочки: последовательно добавляются точки, при этом:
    - если поворот образует правый поворот или коллинеарность, предыдущая точка удаляется;
    - используется проверка знака векторного произведения.
  4. Результат: полученный стек точек является выпуклой оболочкой.

### Код последовательной реализации

```cpp
bool RedkinaAGrahamApproachSEQ::RunImpl() {
  auto pts = GetInput();
  auto res = GrahamScanSeq(std::move(pts));
  if (res.empty() && !GetInput().empty()) {
    res.push_back(GetInput().front());
  }
  GetOutput() = std::move(res);
  return true;
}
```

---

## Схема распараллеливания (MPI)

  В MPI-версии применяется подход с частичными выпуклыми оболочками.

### Основные этапы алгоритма

1. Инициализация:
  - Определяются `rank` и `size`.
  - Исходные данные доступны только на процессе `rank = 0`.

2. Распределение точек:
  - Используется `MPI_Scatterv` для равномерного распределения точек.
  - Создаётся пользовательский тип `MPI_Datatype` для структуры `Point`.

3. Локальное построение оболочки:
  - Каждый процесс применяет алгоритм Грэхема к своему подмножеству точек.
  - Если число точек меньше 3, данные передаются без изменений.

4. Сбор локальных оболочек:
  - Размеры локальных оболочек собираются с помощью `MPI_Gather`.
  - Сами точки — с помощью `MPI_Gatherv`.

5. Финальное построение:
  - Процесс `rank = 0` выполняет алгоритм Грэхема над объединением всех локальных оболочек.
  - Полученная оболочка является глобальным результатом.

6. Рассылка результата
  - Итоговая оболочка рассылается всем процессам с помощью `MPI_Bcast`.


### Код параллельной реализации

```cpp
bool RedkinaAGrahamApproachMPI::RunImpl() {
  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<Point> a_points;
  int n = 0;

  if (rank == 0) {
    a_points = GetInput();
    n = static_cast<int>(a_points.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 3) {
    GetOutput() = (rank == 0 ? a_points : std::vector<Point>{});
    return true;
  }

  MPI_Datatype p_type = MPI_DATATYPE_NULL;
  CreateMpiPointType(&p_type);

  const int base = n / size;
  const int rem = n % size;
  const int l_size = (rank < rem) ? (base + 1) : base;

  std::vector<Point> l_points(l_size);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  InitCountsAndDispls(rank, size, n, counts, displs);

  ScatterPointData(rank, a_points, l_points, counts, displs, p_type);

  std::vector<Point> l_hull =
      (l_size >= 3) ? GrahamScan(std::move(l_points)) : std::move(l_points);

  int l_count = static_cast<int>(l_hull.size());

  std::vector<int> r_counts(size);
  std::vector<int> r_displs(size);

  MPI_Gather(&l_count, 1, MPI_INT, rank == 0 ? r_counts.data() : nullptr,
             1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      r_displs[i] = (i == 0) ? 0 : r_displs[i - 1] + r_counts[i - 1];
      total += r_counts[i];
    }
  }

  std::vector<Point> a_hull_points(total);
  GatherLocalHulls(rank, l_hull, a_hull_points, r_counts, r_displs, p_type);

  std::vector<Point> f_hull = ComputeFinalHull(rank, a_hull_points);

  BroadcastHull(rank, f_hull, p_type);

  GetOutput() = std::move(f_hull);

  MPI_Type_free(&p_type);
  return true;
}
```

---

## Экспериментальные результаты

### Окружение
| Параметр | Значение |
|-----------|-----------|
| Процессор | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics |
| Операционная система | Windows 11 |
| Компилятор | g++ 13.3.0 |
| Тип сборки | Release |
| Число процессов | 2 |

### Проверка корректности
Были проведены функциональные, граничные и расширенные тесты:
  - Минимальные случаи (3 точки);
  - Выпуклый квадрат;
  - Коллинеарные точки;
  - Точки с отрицательными координатами;
  - Случайные большие наборы данных;
  - Проверка корректности выпуклости и принадлежности всех точек оболочке.
  - Последовательная и MPI-версии возвращают эквивалентные результаты.

**Результат:** Все тесты успешно пройдены, последовательная и MPI-версии возвращают одинаковые значения.

### Оценка производительности
  Для оценки производительности использовались тесты с вектором из 1000000 элементов.

**Время выполнения task_run**

| Режим | Процессы | Время, с |
|-------|----------|----------|
| seq   | 1        | 0.232    |
| mpi   | 2        | 0.110    |
| mpi   | 3        | 0.077    |
| mpi   | 4        | 0.058    |

**Время выполнения pipeline**

| Режим | Процессы | Время, с |
|-------|----------|----------|
| seq   | 1        | 0.208    |
| mpi   | 2        | 0.108    |
| mpi   | 3        | 0.075    |
| mpi   | 4        | 0.052    |

**Результат:** Все тесты успешно пройдены.

**Наблюдения:**
  - MPI-версия корректно масштабируется при увеличении числа процессов;
  - Основные накладные расходы связаны с операциями `Scatter/Gather`.

---

## Выводы

  1. **Корректность:** реализованный алгоритм Грэхема корректно строит выпуклую оболочку как в последовательном, так и в параллельном режимах.
  2. **Параллелизм:** подход с построением локальных оболочек и последующим объединением показал свою эффективность и простоту реализации.  
  3. **Масштабируемость:** MPI-версия позволяет обрабатывать большие объёмы данных, однако требует синхронизации на этапе финального объединения. 

---

## Источники

  1. Лекции Сысоева Александра Владимировича.

## Приложения

**common.hpp**
```cpp
#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

struct Point {
  int x{};
  int y{};

  constexpr bool operator==(const Point &other) const noexcept {
    return x == other.x && y == other.y;
  }

  constexpr bool operator!=(const Point &other) const noexcept {
    return !(*this == other);
  }
};

constexpr bool ArePointsEqual(const Point &p1, const Point &p2) noexcept {
  return p1.x == p2.x && p1.y == p2.y;
}

constexpr int CalcCross(const Point &p1, const Point &p2, const Point &p3) noexcept {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p2.y - p1.y) * (p3.x - p1.x));
}

constexpr int CalcDistSq(const Point &p1, const Point &p2) noexcept {
  const int dx = p2.x - p1.x;
  const int dy = p2.y - p1.y;
  return (dx * dx) + (dy * dy);
}

inline Point FindPivotPoint(const std::vector<Point> &points) {
  return *std::ranges::min_element(
      points, [](const Point &a, const Point &b) { return a.y < b.y || (a.y == b.y && a.x < b.x); });
}

using InType = std::vector<Point>;
using OutType = std::vector<Point>;
using TestType = std::tuple<int, std::vector<Point>, std::vector<Point>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace redkina_a_graham_approach
```

**ops_seq.hpp**
```cpp
#pragma once

#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

class RedkinaAGrahamApproachSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaAGrahamApproachSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> GrahamScan(std::vector<Point> points);
};

}  // namespace redkina_a_graham_approach
```

**ops_mpi.hpp**
```cpp
#pragma once

#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_graham_approach {

class RedkinaAGrahamApproachMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() noexcept {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RedkinaAGrahamApproachMPI(const InType &in);

  static std::vector<Point> GrahamScan(std::vector<Point> points);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Point> ComputeFinalHull(int rank, std::vector<Point> &all_hull_points);
};

}  // namespace redkina_a_graham_approach
```

**ops_seq.cpp**
```cpp
#include "redkina_a_graham_approach/seq/include/ops_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"

namespace redkina_a_graham_approach {

RedkinaAGrahamApproachSEQ::RedkinaAGrahamApproachSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaAGrahamApproachSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RedkinaAGrahamApproachSEQ::PreProcessingImpl() {
  return true;
}

namespace {

constexpr bool ComparePolarAngles(const Point &pivot, const Point &a, const Point &b) noexcept {
  const int cross = ((a.x - pivot.x) * (b.y - pivot.y)) -
                    ((a.y - pivot.y) * (b.x - pivot.x));
  if (cross == 0) {
    const int dx1 = a.x - pivot.x;
    const int dy1 = a.y - pivot.y;
    const int dx2 = b.x - pivot.x;
    const int dy2 = b.y - pivot.y;
    return ((dx1 * dx1) + (dy1 * dy1)) < ((dx2 * dx2) + (dy2 * dy2));
  }
  return cross > 0;
}

std::vector<Point> GrahamScanSeq(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  const auto pivot_it = std::ranges::min_element(
      points, [](const Point &a, const Point &b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
      });
  std::swap(points.front(), *pivot_it);
  const Point pivot = points.front();

  std::ranges::sort(points.begin() + 1, points.end(),
                    [&pivot](const Point &a, const Point &b) {
                      return ComparePolarAngles(pivot, a, b);
                    });

  std::vector<Point> hull;
  hull.reserve(points.size());
  for (const auto &p : points) {
    while (hull.size() >= 2 &&
           CalcCross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  return hull;
}

}  // namespace

bool RedkinaAGrahamApproachSEQ::RunImpl() {
  auto pts = GetInput();
  auto res = GrahamScanSeq(std::move(pts));
  if (res.empty() && !GetInput().empty()) {
    res.push_back(GetInput().front());
  }
  GetOutput() = std::move(res);
  return true;
}

bool RedkinaAGrahamApproachSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_graham_approach
```

**ops_mpi.cpp**
```cpp
#include "redkina_a_graham_approach/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "redkina_a_graham_approach/common/include/common.hpp"

namespace redkina_a_graham_approach {

RedkinaAGrahamApproachMPI::RedkinaAGrahamApproachMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RedkinaAGrahamApproachMPI::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool RedkinaAGrahamApproachMPI::PreProcessingImpl() {
  return true;
}

namespace {

inline bool ComparePolarAngle(const Point &pivot, const Point &a, const Point &b) noexcept {
  const int cross = CalcCross(pivot, a, b);
  if (cross == 0) {
    return CalcDistSq(pivot, a) < CalcDistSq(pivot, b);
  }
  return cross > 0;
}

void CreateMpiPointType(MPI_Datatype *p_type) {
  MPI_Type_contiguous(2, MPI_INT, p_type);
  MPI_Type_commit(p_type);
}

void ScatterPointData(int rank, const std::vector<Point> &a_points, std::vector<Point> &l_points,
                      const std::vector<int> &counts, const std::vector<int> &displs,
                      MPI_Datatype p_type) {
  MPI_Scatterv(rank == 0 ? a_points.data() : nullptr, rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, p_type, l_points.data(),
               static_cast<int>(l_points.size()), p_type, 0, MPI_COMM_WORLD);
}

void GatherLocalHulls(int rank, const std::vector<Point> &l_hull, std::vector<Point> &a_hull_points,
                      const std::vector<int> &r_counts, const std::vector<int> &r_displs,
                      MPI_Datatype p_type) {
  MPI_Gatherv(l_hull.data(), static_cast<int>(l_hull.size()), p_type,
              rank == 0 ? a_hull_points.data() : nullptr,
              rank == 0 ? r_counts.data() : nullptr,
              rank == 0 ? r_displs.data() : nullptr, p_type, 0, MPI_COMM_WORLD);
}

void BroadcastHull(int rank, std::vector<Point> &f_hull, MPI_Datatype p_type) {
  int f_size = static_cast<int>(f_hull.size());
  MPI_Bcast(&f_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    f_hull.resize(f_size);
  }

  MPI_Bcast(f_hull.data(), f_size, p_type, 0, MPI_COMM_WORLD);
}

void InitCountsAndDispls(int rank, int size, int n, std::vector<int> &counts, std::vector<int> &displs) {
  if (rank == 0) {
    const int base = n / size;
    const int rem = n % size;

    for (int i = 0; i < size; ++i) {
      counts[i] = (i < rem) ? (base + 1) : base;
    }

    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }
}

}  // namespace

std::vector<Point> RedkinaAGrahamApproachMPI::GrahamScan(std::vector<Point> points) {
  if (points.size() < 3) {
    return points;
  }

  const Point pivot = FindPivotPoint(points);
  std::erase_if(points, [&pivot](const Point &p) { return ArePointsEqual(p, pivot); });

  std::ranges::sort(points, [&pivot](const Point &a, const Point &b) {
    return ComparePolarAngle(pivot, a, b);
  });

  std::vector<Point> hull;
  hull.reserve(points.size() + 1);
  hull.push_back(pivot);
  hull.push_back(points[0]);
  hull.push_back(points[1]);

  for (std::size_t i = 2; i < points.size(); ++i) {
    while (hull.size() >= 2 &&
           CalcCross(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  return hull;
}

std::vector<Point> RedkinaAGrahamApproachMPI::ComputeFinalHull(int rank,
                                                              std::vector<Point> &a_hull_points) {
  if (rank != 0) {
    return {};
  }

  if (a_hull_points.size() >= 3) {
    return GrahamScan(std::move(a_hull_points));
  }

  return std::move(a_hull_points);
}

bool RedkinaAGrahamApproachMPI::RunImpl() {
  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<Point> a_points;
  int n = 0;

  if (rank == 0) {
    a_points = GetInput();
    n = static_cast<int>(a_points.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 3) {
    GetOutput() = (rank == 0 ? a_points : std::vector<Point>{});
    return true;
  }

  MPI_Datatype p_type = MPI_DATATYPE_NULL;
  CreateMpiPointType(&p_type);

  const int base = n / size;
  const int rem = n % size;
  const int l_size = (rank < rem) ? (base + 1) : base;

  std::vector<Point> l_points(l_size);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  InitCountsAndDispls(rank, size, n, counts, displs);

  ScatterPointData(rank, a_points, l_points, counts, displs, p_type);

  std::vector<Point> l_hull =
      (l_size >= 3) ? GrahamScan(std::move(l_points)) : std::move(l_points);

  int l_count = static_cast<int>(l_hull.size());

  std::vector<int> r_counts(size);
  std::vector<int> r_displs(size);

  MPI_Gather(&l_count, 1, MPI_INT, rank == 0 ? r_counts.data() : nullptr,
             1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      r_displs[i] = (i == 0) ? 0 : r_displs[i - 1] + r_counts[i - 1];
      total += r_counts[i];
    }
  }

  std::vector<Point> a_hull_points(total);
  GatherLocalHulls(rank, l_hull, a_hull_points, r_counts, r_displs, p_type);

  std::vector<Point> f_hull = ComputeFinalHull(rank, a_hull_points);

  BroadcastHull(rank, f_hull, p_type);

  GetOutput() = std::move(f_hull);

  MPI_Type_free(&p_type);
  return true;
}

bool RedkinaAGrahamApproachMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_graham_approach
```

