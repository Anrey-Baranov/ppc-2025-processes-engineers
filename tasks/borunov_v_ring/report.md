# Кольцо

- Student: **Борунов Владислав Алексеевич**, group **3823Б1ПР3**
- Technology: **SEQ | MPI**
- Variant: **6**

## 1. Introduction

Работа посвящена реализации и анализу алгоритма передачи сообщения по кольовой топологии. Цель - реализовать последовательную и MPI-реализацию передачи сообщения от источника до цели по кольцу процессов, зафиксировать пройденный путь и проанализировать поведение алгоритма по времени.

## 2. Problem Statement

Определение задачи:

Для заданных параметров входа (номер источника source_rank, номер цели target_rank и некоторое целевое целочисленное значение data) необходимо передать сообщение последовательно по кольцу процессов от source до target, при этом каждый процесс–участник добавляет свою метку (номер ранга) в историю пути (path_history). В конце выполнение задачи должно вернуть вектор пройденных рангов (path).

Формат I/O:

Вход (I): структура { source_rank: int, target_rank: int, data: int }.

Выход (O): вектор std::vector<int> - последовательность рангов, через которые прошло сообщение.

## 3. Последовательная реализация (SEQ)

Последовательная версия вычисляет количество процессов (size), приводит номера source и target к диапазону [0, size-1] и формирует путь по кольцу от source до target инкрементируя индекс по модулю size:

```cpp
std::vector<int> path;
int current = source;
path.push_back(current);
while (current != target) {
  current = (current + 1) % size;
  path.push_back(current);
}
```

Для имитации вычислительной нагрузки в цикле выполняется набор операций в течение фиксированного времени 800 ms.

Возврат: GetOutput() = path;.

## 4. Parallelization Scheme (MPI)

Идея распараллеливания:

- Используется кольцевая топология MPI каждый процесс знает prev и next.
- Только участники, попадающие в интервал от source до target по кольцу, участвуют в пересылке сообщения и сформировании пути включая source и target.
- source инициализирует path_history включает свой ранг, выполняет задержку и отправляет размер пути, сами элементы пути и сопутствующие данные следующему процессу.
- Каждый участник получает пакет от предыдущего, добавляет свой ранг в историю, выполняет задержку и либо завершает, либо пересылает дальше.

Коммуникационные операции:

- MPI_Send / MPI_Recv - отправка размера и массива пути, а также сопутствующего data.
- MPI_Barrier и MPI_Comm_dup используются для синхронизации и изоляции коммуникаций.

Особенности:

- Передача представляет собой последовательную цепочку зависимых передач: следующий шаг ожидает данных от предыдущего.

## 5. Implementation Details

### 5.1 Структура кода

Файлы:

- `common/include/common.hpp` - общие типы и структура входа/выхода.
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация.
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - MPI-реализация.
- `tests/functional/main.cpp` - функциональные тесты.
- `tests/performance/main.cpp` - тесты производительности.

Ключевые методы:

- ComputeIsParticipant() - проверяет, участвует ли ранк в пути между source и target.
- HandleSource() - обработка на стороне source: инициализация path_history, задержка, отправка пакета следующему.
- HandleParticipant() - обработка у промежуточного участника: приём, задержка, добавление в path_history, пересылка дальше или завершение.

### 5.2 Важные решения при реализации

- Все коммуникации выполняются точечно (point-to-point), полный путь передаётся последовательно.
- Для симуляции затрат на обработку использована небольшая вычислительная задержка AddDelay() 200 ms в MPI-реализации и более длительная в SEQ 800 ms.
- Функции защищают от неверных размеров и некорректных значений path_size при приёме.

## 6. Experimental Setup

### Аппаратное обеспечение

- CPU: i7-12650H
- Ядра: 10
- Потоки: 16
- ОЗУ: 16 ГБ
- ОС: Windows 11

### Программное обеспечение

- Компилятор: MSVC 14.44
- Сборка: Release

### Генерация тестовых сценариев

- Тестовые сценарии задают ring_size через mpirun -n N.

## 7. Results and Discussion

### 7.1 Correctness

- Функциональные тесты проходят успешно: GetOutput() возвращает корректную последовательность рангов от source до target по модулю размера кольца.
- Пограничные случаи: source == target, source > target склейка по модулю и случаи с world_size == 1 корректно обработаны.

### 7.2 Performance

| Режим | Процессов | Время, сек | 
|-------|-----------|------------|
| seq   | 1         | 0.8010     | 
|-------|-----------|------------|
| mpi   | 4         | 0.8254     |
|-------|-----------|------------| 
| mpi   | 64        | 12.816     |
|-------|-----------|------------|
| mpi   | 100       | 20.043     |
|-------|-----------|------------|

## 8. Conclusions

- Реализована корректная схема последовательной передачи по кольцу: сообщение проходит по всем ранкам от source к target с накоплением истории пути.
- MPI-реализация корректно обрабатывает границы и случаи source==target и склейки при source>target.

## 9. References

- Microsoft. Microsoft MPI Documentation. https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- Сысоев А. В. Лекции по параллельному программированию. — Н. Новгород: ННГУ, 2025.

## Appendix

Ключевые фрагменты реализации:

```cpp
// Фрагмент SEQ: формирование пути
std::vector<int> path;
int current = source;
path.push_back(current);
while (current != target) {
  current = (current + 1) % size;
  path.push_back(current);
}
GetOutput() = path;
```

```cpp
// Фрагменты MPI: отправка на source
void HandleSource(BorunovVRingMPI *self, MPI_Comm ring_comm, int ring_rank, int next_rank, int target, int data) {
  std::vector<int> path_history;
  path_history.push_back(ring_rank);
  AddDelay(); // имитация работы
  if (ring_rank == target) {
    self->GetOutput() = std::move(path_history);
    return;
  }
  int path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  if (path_size > 0) {
    MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  }
  MPI_Send(&data, 1, MPI_INT, next_rank, 2, ring_comm);
}
```

```cpp
// Фрагмент MPI: обработка промежуточного участника
void HandleParticipant(BorunovVRingMPI *self, MPI_Comm ring_comm, int prev_rank, int next_rank, int ring_rank, int target) {
  int path_size = 0;
  MPI_Status status;
  MPI_Recv(&path_size, 1, MPI_INT, prev_rank, 0, ring_comm, &status);
  std::vector<int> path_history(static_cast<std::size_t>(path_size));
  if (path_size > 0) {
    MPI_Recv(path_history.data(), path_size, MPI_INT, prev_rank, 1, ring_comm, &status);
  }
  int received_data = 0;
  MPI_Recv(&received_data, 1, MPI_INT, prev_rank, 2, ring_comm, &status);
  AddDelay();
  path_history.push_back(ring_rank);
  if (ring_rank == target) {
    self->GetOutput() = path_history;
    return;
  }
  path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  MPI_Send(&received_data, 1, MPI_INT, next_rank, 2, ring_comm);
}
```

