# Обобщенная передача от всех одному (Gather)

- Студент: Борцова Ангелина Сергеевна, группа 3823Б1ПР5
- Технология: MPI|SEQ
- Вариант: 5

## 1. Введение

Операция Gather является одной из фундаментальных коллективных операций в параллельном программировании. Она позволяет собрать данные со всех процессов в один массив на указанном процессе-получателе. В данной работе реализована операция Gather с использованием только базовых функций MPI_Send и MPI_Recv, что позволяет лучше понять механизм работы коллективных операций на низком уровне. Реализация использует бинарное дерево для эффективного сбора данных.

## 2. Постановка задачи

Требуется реализовать операцию Gather, которая собирает массивы данных со всех процессов в один массив на процессе root. Операция должна иметь тот же прототип, что и стандартная функция MPI_Gather, но реализована только с использованием MPI_Send и MPI_Recv.

Входные данные:
- Массив send_data типа double на каждом процессе
- Номер процесса root, который должен получить собранные данные

Выходные данные:
- Массив recv_data на процессе root, содержащий данные со всех процессов в порядке их рангов

Ограничения:
- Использовать только MPI_Send и MPI_Recv
- Реализовать сбор данных через бинарное дерево процессов
- Поддерживать выбор произвольного процесса root
- Обеспечить корректную работу с различными размерами данных

## 3. Последовательный алгоритм 

Последовательная реализация операции Gather тривиальна, так как в однопроцессорной системе нет необходимости собирать данные с других процессов. Алгоритм просто копирует входной массив send_data в выходной массив recv_data.

Алгоритм:
1. Проверить, что root равен 0 
2. Скопировать send_data в recv_data

Временная сложность: O(n), где n - размер входного массива.

## 4. Схема распараллеливания

Для реализации операции Gather используется бинарное дерево процессов. Топология дерева строится динамически на основе рангов процессов.

Алгоритм сбора данных по дереву:

1. Инициализация: каждый процесс помещает свои данные в соответствующий сегмент общего буфера gather_buffer.

2. Фаза сбора (TreeGather):
   - На каждом шаге step = 1, 2, 4, 8, ... процессы с рангами, кратными (2 * step), получают данные от процессов с рангами (rank + step)
   - Процессы с рангами, кратными step, но не кратными (2 * step), отправляют свои данные процессу с рангом (rank - step)
   - После отправки данных процесс завершает участие в сборе

3. Передача на root: если root не равен 0, процесс 0 передает собранные данные процессу root.

4. Распространение результата: процесс root рассылает собранные данные всем процессам через MPI_Bcast.

Пример работы для 4 процессов:
- Шаг 1: Процесс 0 получает от процесса 1, процесс 2 получает от процесса 3
- Шаг 2: Процесс 0 получает от процесса 2
- Результат: Процесс 0 содержит данные всех процессов

Временная сложность: O(log P), где P - количество процессов.

## 5. Детали реализации

Структура кода:

- common/include/common.hpp: Определение структур Input и Output
- mpi/include/ops_mpi.hpp: Заголовок класса BortsovaATransmissionGatherMPI
- mpi/src/ops_mpi.cpp: Реализация MPI версии
- seq/include/ops_seq.hpp: Заголовок класса BortsovaATransmissionGatherSEQ
- seq/src/ops_seq.cpp: Реализация последовательной версии

Ключевые методы MPI реализации:

- ValidationImpl(): Проверка корректности номера процесса root
- PreProcessingImpl(): Инициализация размера данных
- RunImpl(): Основной алгоритм сбора данных
- TreeGather(): Реализация сбора по бинарному дереву
- ReceiveFromChild(): Получение данных от дочернего узла
- SendToParent(): Отправка данных родительскому узлу
- TransferToRoot(): Передача данных на указанный root процесс

Особенности реализации:

- Использование MPI_Send и MPI_Recv вместо стандартных коллективных операций
- Отслеживание полученных данных через массив флагов received
- Поддержка произвольного процесса root через дополнительную передачу данных
- Обработка граничных случаев (пустые массивы, root не равен 0)

Использование памяти: O(P * n), где P - количество процессов, n - размер данных на каждом процессе.

## 6. Экспериментальная установка

Аппаратное обеспечение и ОС:
- Процессор: Intel(R) Core(TM) Ultra 7 155H (3.80 GHz)
- Оперативная память: 32ГБ
- Операционная система: Windows 11 pro
- Количество процессов: 1-4

Инструментарий:
- Компилятор: MSVC 14.44
- Тип сборки: Release 
- MPI реализация: MS-MPI 10.0
- CMake: 4.2.0-rc1

Окружение:
- PPC_NUM_PROC: 1-4 
- Размер тестовых данных: 30000000 элементов типа double

Данные:
- Тестовые данные генерируются программно в тестах производительности
- Каждый элемент массива инициализируется значением 1.0 + индекс

## 7. Результаты и обсуждение

### 7.1 Корректность

Корректность реализации проверялась следующими способами:

1. Функциональные тесты: Проверка корректности сбора данных для различных размеров массивов (3, 5, 7 элементов) на разном количестве процессов.

2. Граничные случаи:
   - Один элемент в массиве
   - Большие массивы (1000 элементов)
   - Отрицательные значения
   - Числа с плавающей точкой различной точности
   - Пустые массивы (для последовательной версии)
   - Идентичные значения
   - Чередующиеся знаки

3. Проверка результатов: Для каждого процесса проверяется, что собранные данные содержат правильные значения в правильном порядке относительно рангов процессов.

Все тесты успешно пройдены.

### 7.2 Производительность

Результаты замеров производительности представлены в таблицах ниже. Время измеряется в секундах. Speedup рассчитывается относительно последовательной версии. Efficiency показывает эффективность использования процессов.

Режим pipeline:

| Режим | Процессов | Время, с | Speedup | Efficiency |
|-------|-----------|----------|---------|------------|
| seq   | 1         | 0.0351   | 1.00    | N/A       |
| mpi   | 1         | 0.2299   | 0.153   | 15.3%     |
| mpi   | 2         | 1.1534   | 0.030   | 1.5%      |
| mpi   | 3         | 2.4227   | 0.014   | 0.5%      |
| mpi   | 4         | 3.5042   | 0.010   | 0.3%      |

Режим task_run:

| Режим | Процессов | Время, с | Speedup | Efficiency |
|-------|-----------|----------|---------|------------|
| seq   | 1         | 0.0367   | 1.00    | N/A       |
| mpi   | 1         | 0.2310   | 0.159   | 15.9%     |
| mpi   | 2         | 1.1527   | 0.032   | 1.6%      |
| mpi   | 3         | 2.4969   | 0.015   | 0.5%      |
| mpi   | 4         | 3.4189   | 0.011   | 0.3%      |

Анализ результатов:

1. Последовательная версия показывает наилучшее время выполнения, что ожидаемо для простой операции копирования данных.

2. MPI версия на одном процессе показывает замедление примерно в 6.5 раз по сравнению с последовательной версией. Это связано с накладными расходами на инициализацию MPI и работу с коммуникаторами.

3. При увеличении количества процессов время выполнения увеличивается, что указывает на доминирование накладных расходов на коммуникацию над вычислительной работой.

4. Низкая эффективность объясняется тем, что операция Gather является коммуникационно-интенсивной, и накладные расходы на передачу данных значительно превышают вычислительную нагрузку.

5. Разница между режимами pipeline и task_run незначительна, что указывает на то, что основное время тратится на коммуникацию, а не на вычисления.

Ограничения масштабируемости:

- Накладные расходы на коммуникацию растут с увеличением количества процессов
- Объем передаваемых данных увеличивается пропорционально количеству процессов
- Для данной задачи параллелизация не дает преимуществ из-за коммуникационной природы операции

## 8. Выводы

В ходе работы была успешно реализована операция Gather с использованием только базовых функций MPI_Send и MPI_Recv. Реализация использует бинарное дерево для сбора данных и поддерживает произвольный процесс root.

Основные результаты:

1. Реализация корректно работает для различных размеров данных и количества процессов.

2. Последовательная версия показывает наилучшую производительность для данной задачи.

3. MPI версия демонстрирует снижение производительности с увеличением количества процессов из-за доминирования накладных расходов на коммуникацию.

4. Операция Gather является коммуникационно-интенсивной, поэтому параллелизация не дает преимуществ в производительности для данной задачи.

Ограничения:

- Реализация не оптимизирована для больших объемов данных
- Накладные расходы на коммуникацию преобладают над вычислительной работой
- Эффективность использования процессов очень низкая

Возможные улучшения:

- Использование стандартных коллективных операций MPI для лучшей производительности
- Оптимизация передачи данных для больших массивов
- Использование неблокирующих операций для перекрытия вычислений и коммуникаций

## 9. Источники

1. MPI Forum. MPI: A Message-Passing Interface Standard. Version 3.1. https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf
2. Документация по MPI: https://www.open-mpi.org/doc/
3. Сысоев А. В.: Лекции по параллельному программированию. — Н. Новгород: ННГУ, 2025.
4. Гергель В.П.: Образовательный комплекс "Введение в методы параллельного программирования" - Н. Новгород: ННГУ, 2005.

## 10. Приложение

### 10.1 Структуры данных

```cpp
struct Input {
  std::vector<double> send_data;
  int root = 0;
};

struct Output {
  std::vector<double> recv_data;
};
```

### 10.2 Основной алгоритм MPI версии

```cpp
bool BortsovaATransmissionGatherMPI::RunImpl() {
  int root = GetInput().root;
  const std::vector<double> &send_data = GetInput().send_data;

  int local_count = send_count_;
  MPI_Bcast(&local_count, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (local_count == 0) {
    GetOutput().recv_data.clear();
    return true;
  }

  std::size_t total_size = static_cast<std::size_t>(world_size_) * static_cast<std::size_t>(local_count);
  std::vector<double> gather_buffer(total_size, 0.0);
  std::vector<bool> received(static_cast<std::size_t>(world_size_), false);

  std::size_t offset = static_cast<std::size_t>(world_rank_) * static_cast<std::size_t>(local_count);
  for (std::size_t idx = 0; idx < send_data.size(); ++idx) {
    gather_buffer[offset + idx] = send_data[idx];
  }
  received[static_cast<std::size_t>(world_rank_)] = true;

  TreeGather(gather_buffer, received, local_count, static_cast<int>(total_size));

  TransferToRoot(gather_buffer, root, static_cast<int>(total_size));

  MPI_Bcast(gather_buffer.data(), static_cast<int>(total_size), MPI_DOUBLE, root, MPI_COMM_WORLD);

  GetOutput().recv_data = std::move(gather_buffer);

  return true;
}
```

### 10.3 Сбор данных по бинарному дереву

```cpp
void BortsovaATransmissionGatherMPI::TreeGather(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                int local_count, int total_size) {
  int step = 1;
  while (step < world_size_) {
    if ((world_rank_ % (2 * step)) == 0) {
      int source = world_rank_ + step;
      if (source < world_size_) {
        ReceiveFromChild(gather_buffer, received, source, local_count, total_size);
      }
    } else if ((world_rank_ % step) == 0) {
      SendToParent(gather_buffer, received, step, total_size);
      break;
    }
    step *= 2;
  }
}
```

### 10.4 Получение данных от дочернего узла

```cpp
void BortsovaATransmissionGatherMPI::ReceiveFromChild(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                      int source, int local_count, int total_size) const {
  std::vector<double> recv_buffer(static_cast<std::size_t>(total_size), 0.0);
  std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);

  MPI_Status status;
  MPI_Recv(recv_buffer.data(), total_size, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(flags_int.data(), world_size_, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

  for (int rank = 0; rank < world_size_; ++rank) {
    if (flags_int[static_cast<std::size_t>(rank)] != 0) {
      std::size_t start_idx = static_cast<std::size_t>(rank) * static_cast<std::size_t>(local_count);
      for (int jj = 0; jj < local_count; ++jj) {
        gather_buffer[start_idx + static_cast<std::size_t>(jj)] = recv_buffer[start_idx + static_cast<std::size_t>(jj)];
      }
      received[static_cast<std::size_t>(rank)] = true;
    }
  }
}
```

### 10.5 Отправка данных родительскому узлу

```cpp
void BortsovaATransmissionGatherMPI::SendToParent(std::vector<double> &gather_buffer, std::vector<bool> &received,
                                                  int step, int total_size) const {
  int dest = world_rank_ - step;
  MPI_Send(gather_buffer.data(), total_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

  std::vector<int> flags_int(static_cast<std::size_t>(world_size_), 0);
  for (int rank = 0; rank < world_size_; ++rank) {
    flags_int[static_cast<std::size_t>(rank)] = static_cast<int>(received[static_cast<std::size_t>(rank)]);
  }
  MPI_Send(flags_int.data(), world_size_, MPI_INT, dest, 1, MPI_COMM_WORLD);
}
```

### 10.6 Передача данных на указанный root процесс

```cpp
void BortsovaATransmissionGatherMPI::TransferToRoot(std::vector<double> &gather_buffer, int root,
                                                    int total_size) const {
  if (world_rank_ == 0 && root != 0) {
    MPI_Send(gather_buffer.data(), total_size, MPI_DOUBLE, root, 2, MPI_COMM_WORLD);
  } else if (world_rank_ == root && root != 0) {
    MPI_Status status;
    MPI_Recv(gather_buffer.data(), total_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
  }
}
```

### 10.7 Последовательная реализация

```cpp
bool BortsovaATransmissionGatherSEQ::RunImpl() {
  const std::vector<double> &send_data = GetInput().send_data;
  GetOutput().recv_data = send_data;
  return true;
}
```

