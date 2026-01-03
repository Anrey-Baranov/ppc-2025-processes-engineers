# Отчёт по лабораторной работе 
## Топологии сетей передачи данных: Кольцо (Ring)
## Вариант: 7

**Студент:** Ермаков Алексей Викторович, группа 3823Б1ПР3  
**Преподаватели:** Сысоев А. В., Оболенский А. А., Нестеров А.

---

## 1. Введение

В распределенных системах топология определяет способ связи между узлами. Топология «Кольцо» (Ring) подразумевает, что каждый узел соединен ровно с двумя другими, образуя замкнутую цепь. Передача сообщения от источника к пункту назначения может осуществляться в двух направлениях: по часовой стрелке или против неё.

**Цель работы:** реализовать передачи данных в кольцевой топологии, обеспечив выбор кратчайшего пути и корректную передачу вместе с историей маршрута (path).

В работе реализованы:
- **Последовательная версия (SEQ):** используя MPI_Cart_Create и MPI_Graph_Create
- **Параллельная версия (MPI):** передача вручную, используя MPI_Send/MPI_Recv

---

## 2. Постановка задачи

**Входные данные:**
- `source`: идентификатор узла-отправителя.
- `dest`: идентификатор узла-получателя.
- `data`: вектор целых чисел (полезная нагрузка).

**Выходные данные:**
- Вектор `path`, содержащий последовательность всех узлов, через которые прошло сообщение (включая источник и цель).

**Ключевые требования:**
1. Выбор кратчайшего пути между `source` и `dest`.
2. Передача данных только между соседними узлами.
3. Корректная работа при `source == dest`.

---

## 3. Базовый алгоритм (последовательный)

В последовательной реализации (SEQ) для моделирования кольцевой структуры используется механизм декартовых топологий MPI:
1. Создается одномерная декартова топология с помощью `MPI_Cart_create` с параметром `periods = {1}`, что обеспечивает замкнутость (кольцо).
2. Определение соседей (левого и правого) происходит через `MPI_Cart_shift`.
3. Алгоритм вычисляет кратчайшее расстояние в обоих направлениях и выбирает оптимальное.
4. Выполняется последовательный обход узлов с имитацией передачи данных и формированием вектора `path`.

---

## 4. Схема параллелизации (MPI)

Параллельная версия реализует физическую передачу сообщений между процессами-ранками.

### Логика передачи (Point-to-Point)
- **Источник (Source):** Инициирует передачу. Отправляет данные (`payload`) и текущий маршрут соседу в выбранном направлении.
- **Транзитные узлы:** Принимают данные от предыдущего соседа (`prev`), добавляют свой ранк в `path` и пересылают следующему (`next`).
- **Получатель (Dest):** Принимает финальное сообщение. После формирования пути выполняется `MPI_Bcast`, чтобы результат был доступен на всех узлах.

---

## 5. Детали реализации

**Основные файлы:**
- `ops_mpi.cpp` — MPI реализация.
- `ops_seq.cpp` — SEQ реализация.
- `common.hpp` — структуры данных проекта.

**Основные методы:**
- `ValidationImpl()`: проверка корректности входных параметров.
- `RunImpl()`: выполнение алгоритма (передача по кольцу).

---

## 6. Экспериментальная установка

**Оборудование:**
- CPU: Ryzen 5 1600 (3.8 GHz)
- RAM: 8 GB
- OS: Windows 11

**Программное обеспечение:**
- MS-MPI 10.0
- CMake 4.2.0-rc1
- Google Test
- Режим сборки: Release

---

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверялась на наборе функциональных тестов (`RingTests`). Все тесты пройдены успешно, включая специфические случаи передачи через "границу" кольца (например, от последнего ранка к нулевому).

### 7.2 Производительность
Тестирование проводилось при передаче данных объемом **40 000 000** элементов.

| Режим          | Процессы | Время (сек) | Ускорение |
|----------------|----------|-------------|-----------|
| SEQ (task_run) | 1        | 0.4204      | 1.00      |
| MPI (task_run) | 2        | 0.0388      | 10.83     |
| MPI (task_run) | 4        | 0.1234      | 3.40      |
| MPI (task_run) | 8        | 0.2455      | 1.71      |
| MPI (task_run) | 12       | 0.5563      | 0.75      |

**Анализ масштабируемости:**
При увеличении числа процессов (например, до 12) наблюдается рост времени выполнения по сравнению с MPI на 2 процессах. Это обусловлено спецификой топологии «Кольцо»:
1. **Последовательная природа:** Сообщение не может «перепрыгнуть» через узлы. Чем больше процессов в кольце, тем больше промежуточных пересылок (`MPI_Send` / `MPI_Recv`) необходимо выполнить, если источник и цель находятся далеко друг от друга.
2. **Сетевые задержки (Latency):** Каждая промежуточная передача добавляет накладные расходы на инициализацию сообщения.
3. **Объем данных:** При 40 млн элементов затраты на копирование буферов в памяти при каждой передаче между 12 ранками начинают доминировать над временем вычислений.

---

## 8. Выводы
- Реализована модель сетевой топологии «Кольцо» с выбором кратчайшего пути.
- Код успешно прошел все тесты на корректность и соответствует Clang-Tidy.
- Экспериментально подтверждено, что для данной топологии увеличение числа узлов может приводить к замедлению из-за увеличения количества транзитных пересылок.

---

## 9. Список источников

1. MPI Forum. MPI: A Message-Passing Interface Standard, Version 4.0, 2021.
2. Microsoft MPI Documentation.
3. Документация преподавателей: https://learning-process.github.io/parallel_programming_slides/
4. Лекции Сысоев А. В., Оболенский А. А., Нестеров А., ННГУ, 2025.

---

# Приложение

## Фрагмент кода (RunImpl MPI)

```cpp
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  std::vector<int> payload = input.data;

  const int src = ((input.source % size) + size) % size;
  const int dst = ((input.dest % size) + size) % size;

  std::vector<int> path;

  const int cw_dist = (dst - src + size) % size;
  const int cc_dist = (src - dst + size) % size;
  const bool clockwise = (cw_dist <= cc_dist);
  const int steps = clockwise ? cw_dist : cc_dist;

  const int next = clockwise ? (rank + 1) % size : (rank - 1 + size) % size;
  const int prev = clockwise ? (rank - 1 + size) % size : (rank + 1) % size;

  const int dist_from_src = clockwise ? (rank - src + size) % size : (src - rank + size) % size;

  if (src == dst) {
    if (rank == src) {
      path = {src};
    }
  } else {
    if (rank == src) {
      path = {src};
      const int path_sz = static_cast<int>(path.size());
      const int data_sz = static_cast<int>(payload.size());

      MPI_Send(payload.data(), data_sz, MPI_INT, next, 100, MPI_COMM_WORLD);
      MPI_Send(&path_sz, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
      MPI_Send(path.data(), path_sz, MPI_INT, next, 1, MPI_COMM_WORLD);
    } else if (dist_from_src > 0 && dist_from_src <= steps) {
      int path_sz = 0;
      const int data_sz = static_cast<int>(payload.size());

      MPI_Recv(payload.data(), data_sz, MPI_INT, prev, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&path_sz, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      path.resize(static_cast<size_t>(path_sz));
      MPI_Recv(path.data(), path_sz, MPI_INT, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      path.push_back(rank);

      if (rank != dst) {
        const int next_path_sz = static_cast<int>(path.size());
        MPI_Send(payload.data(), data_sz, MPI_INT, next, 100, MPI_COMM_WORLD);
        MPI_Send(&next_path_sz, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Send(path.data(), next_path_sz, MPI_INT, next, 1, MPI_COMM_WORLD);
      }
    }
  }

  int final_path_sz = 0;
  if (rank == dst) {
    final_path_sz = static_cast<int>(path.size());
  }

  MPI_Bcast(&final_path_sz, 1, MPI_INT, dst, MPI_COMM_WORLD);
  if (rank != dst) {
    path.resize(static_cast<size_t>(final_path_sz));
  }
  MPI_Bcast(path.data(), final_path_sz, MPI_INT, dst, MPI_COMM_WORLD);

  GetOutput() = path;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
```

## Фрагмент кода (RunImpl SEQ)

```cpp
  int w_rank = 0;
  int w_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &w_size);

  const int s = s_idx % w_size;
  const int d = d_idx % w_size;
  auto payload = task->GetInput().data;

  int dims[1] = {w_size};
  int periods[1] = {1};
  MPI_Comm ring_comm = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &ring_comm);

  int l_peer = 0;
  int r_peer = 0;
  MPI_Cart_shift(ring_comm, 0, 1, &l_peer, &r_peer);

  const int r_dist = (d - s + w_size) % w_size;
  const int l_dist = (s - d + w_size) % w_size;

  bool move_r = false;
  if (r_dist <= l_dist) {
    move_r = true;
  }

  int nxt = 0;
  int prv = 0;
  int steps_total = 0;
  if (move_r) {
    nxt = r_peer;
    prv = l_peer;
    steps_total = r_dist;
  } else {
    nxt = l_peer;
    prv = r_peer;
    steps_total = l_dist;
  }

  std::vector<int> path;
  if (s == d) {
    if (w_rank == s) {
      path = {s};
    }
  } else {
    int my_dist = 0;
    if (move_r) {
      my_dist = (w_rank - s + w_size) % w_size;
    } else {
      my_dist = (s - w_rank + w_size) % w_size;
    }

    if (w_rank == s) {
      path = {s};
      MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, nxt, 100, ring_comm);
      const int psz = static_cast<int>(path.size());
      MPI_Send(&psz, 1, MPI_INT, nxt, 10, ring_comm);
      MPI_Send(path.data(), psz, MPI_INT, nxt, 11, ring_comm);
    } else if (my_dist > 0 && my_dist <= steps_total) {
      MPI_Recv(payload.data(), static_cast<int>(payload.size()), MPI_INT, prv, 100, ring_comm, MPI_STATUS_IGNORE);
      int in_sz = 0;
      MPI_Recv(&in_sz, 1, MPI_INT, prv, 10, ring_comm, MPI_STATUS_IGNORE);
      path.resize(static_cast<size_t>(in_sz));
      MPI_Recv(path.data(), in_sz, MPI_INT, prv, 11, ring_comm, MPI_STATUS_IGNORE);
      path.push_back(w_rank);
      if (w_rank != d) {
        MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, nxt, 100, ring_comm);
        const int out_sz = static_cast<int>(path.size());
        MPI_Send(&out_sz, 1, MPI_INT, nxt, 10, ring_comm);
        MPI_Send(path.data(), out_sz, MPI_INT, nxt, 11, ring_comm);
      }
    }
  }

  int total_len = 0;
  if (w_rank == d) {
    total_len = static_cast<int>(path.size());
  }
  MPI_Bcast(&total_len, 1, MPI_INT, d, ring_comm);
  if (w_rank != d) {
    path.resize(static_cast<size_t>(total_len));
  }
  MPI_Bcast(path.data(), total_len, MPI_INT, d, ring_comm);

  task->GetOutput() = path;
  MPI_Comm_free(&ring_comm);
  return true;
```
