# Линейка

- Студент: Табалаев Антон Максимович
- Технология: SEQ | MPI
- Вариант: 6

## 1. Введение 

Топология в MPI – это механизм, который сопоставляет процессам некоторого коммуникатора альтернативную (логическую) схему адресации. При этом топологии являются виртуальными, то есть они не связаны с физической топологией коммуникационной сети. Топология используется для более удобного обозначения процессов, что позволяет приблизить структуру параллельной программы к модели математического алгоритма. Кроме того, она может применяться системой для оптимизации распределения процессов по физическим процессорам путём изменения порядка их нумерации в коммуникаторе.

## 2. Постановка задачи

**Цель работы:**
Реализовать MPI-параллельную версию передачи данных по линейной топологии между заданными процессами.

**Определение задачи:**
Дано:
- *sender* - ранг процесса отпрвителся.
- *receiver* - ранг процесса получателся.
- *data* - вектор данных для передачи.

Необходимо передать вектор данных по цепочке процессов от `sender` к `receiver`.

Ограничения:
- Не допускается использование `MPI_Cart_Create` и `MPI_Graph_Create`.
- Передача данных должна работать между любыми процессами.
- Отправитель и получатель могут совпадать.

## 3. Алгоритм (Последовательная версия)

В последовательной версии алгоритма отсутствует распределение данных между процессами. Алгоритм выполняется в одном потоке и используется для проверки корректности работы и сравнения с параллельной реализацией.

Алгоритм:
1. Получить входные данные `sender`, `receiver` и вектор `data`.
2. Проверить корректность входных данных (отрицательные значения sender и receiver недопустимы, вектор data не должен быть пустым).
3. Скопировать входной вектор `data` в выходной вектор.

### Код последовательной версии алгоритма
```
bool TabalaevALinearTopologySEQ::ValidationImpl() {
  auto &sender = std::get<0>(GetInput());
  auto &receiver = std::get<1>(GetInput());
  auto &data = std::get<2>(GetInput());
  if (sender < 0 || receiver < 0) {
    return false;
  }
  return !data.empty();
}
bool TabalaevALinearTopologySEQ::RunImpl() {
  auto &data = std::get<2>(GetInput());
  GetOutput() = data;
  return true;
}
```

## 4. Схема распараллеливания 

В MPI-версии передача данных выполняется по цепочке процессов, обеспечивая соблюдение линейной топологии.

- **Проверка входных данных:** Процесс отправитель `sender` проверяет корректность входных данных, а затем рассылает флаг валидности через `MPI_Bcast`.
- **Инициализация:** Все процессы запускаются в коммуникаторе MPI_COMM_WORLD, определяется общее количество процессов и ранг текущего процесса.
- **Маршрутизация сообщения:** 
  - **Определяется направление передачи данных:** Если `sender < receiver`, то передача идёт по возрастанию рангов, иначе по убыванию. Если же `sender == receiver` то данные возвращаются без передачи.
  - **Определение соседей:** Каждый процесс определяет своих левого и правого соседей (для крайних случаев один из соседей равен `MPI_PROC_NULL`).
- **Передача данных:** 
  - Процесс отправитель с рангом `sender` определяет своего соседа, а затем отправляет ему размер и сам вектор через `MPI_Send`.
  - Промежуточные процессы сначала определяют находятся ли они на пути передачи (сравнивают свой ранг с рангами отправителя и получателя с учётом направления). Если находятся, то получают размер и данные от предыдущего в цепочке соседа через `MPI_Recv`. Далее этот процесс пересылает размер и данные следующему в цепочке.
  - Процесс получатель `receiver` принимает данные от предыдущего процесса в цепочке.
- **Формирование результата:** Процесс получатель рассылает полученные данные всем процессам через `MPI_Bcast`.

Реализация параллельного алгоритма представлена в Приложении 1.

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
- Используют различные комбинации отправителя (sender) и получателя (receiver), включая случаи sender == receiver и sender ≠ receiver (например, sender = 0, receiver = 5).
- Размер входного вектора данных варьируется от 50 до 200 элементов.
- Значения элементов вектора вычисляются по формуле `i * i + 3`.
- Проверяется корректность копирования входного вектора в выходной.
2. Тесты производительности:
- Используют вектор размером 10_000_000 элементов.
- Значения элементов вычисляются по формуле `i * i + 2`.
- Используют различные комбинации отправителя (sender) и получателя (receiver), включая случаи sender == receiver и sender ≠ receiver (например, sender = 0, receiver = 5).

### Производительность

| Режим выполнения | Sender | Receiver | Количество процессов | Время выполнения (с) |
|------------------|--------|----------|----------------------|----------------------|
| **MPI** (1 поцесс) | | | | |
| pipeline | 0 | 0 | 1 | 0.00895752 |
| task_run | 0 | 0 | 1 | 0.00241518 |
| **MPI** (2 поцесса) | | | | |
| pipeline | 0 | 1 | 2 | 0.02877076 |
| task_run | 0 | 1 | 2 | 0.02238150 |
| **MPI** (4 поцесса) | | | | |
| pipeline | 0 | 3 | 4 | 0.05896450 |
| task_run | 0 | 3 | 4 | 0.05161030 |
| **MPI** (6 поцессов) | | | | |
| pipeline | 0 | 5 | 6 | 0.08926704 |
| task_run | 0 | 5 | 6 | 0.07805980 |

## 6. Результаты

1. Результаты функционального тестирования
- Все функциональные тесты успешно пройдены.
- Последовательная и MPI-реализации алгоритма корректно обрабатывают входные данные для различных комбинаций отправителя и получателя.

2. Результаты сравнения производительности
- В ходе тестирования было установлено, что MPI-реализация алгоритма на 2 и более процессах демонстрирует большее время выполнения по сравнению с запуском на одном процессе.
- С увеличением количества процессов время выполнения возрастает из-за расходов на передачу данных между процессами.

## 7. Выводы

1. Разработанные алгоритмы корректно выполняют передачу данных в линейной топологии от любого процесса к любому другому, что подтверждается успешным прохождением всех тестов.
2. Алгоритмы не рассчитаны на повышения скорости вычислений, а служат для демонстрации правильной маршрутизации сообщений.

## 8. Источники

1. Сысоев А. В. Лекции по курсу «Параллельное программирование для кластерных систем».
2. Официальная документация Microsoft MPI — https://learn.microsoft.com/ru-ru/message-passing-interface
3. Документация Open MPI — https://www.open-mpi.org/doc/
4. C++ Reference — https://en.cppreference.com
5. MPI - Message Passing Interface — https://parallel.ru/sites/default/files/info/parallel/antonov/topology.pdf

## 9. Приложение

### Приложение 1 (Код параллельной версии алгоритма)

```
bool TabalaevALinearTopologyMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int sender = std::get<0>(GetInput());

  int validation = 1;
  if (world_rank == sender) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    auto receiver = std::get<1>(GetInput());
    auto data = std::get<2>(GetInput());

    if ((sender < 0 || sender >= world_size) || (receiver < 0 || receiver >= world_size) || data.empty()) {
      validation = 0;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, sender, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevALinearTopologyMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool TabalaevALinearTopologyMPI::RunImpl() {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto sender = std::get<0>(GetInput());
  auto receiver = std::get<1>(GetInput());

  if (sender == receiver) {
    GetOutput() = std::get<2>(GetInput());
    return true;
  }

  int left = GetLeft(world_rank);
  int right = GetRight(world_rank, world_size);
  int direction = GetDirection(sender, receiver);

  std::vector<int> local_buff;

  if (world_rank == sender) {
    ProcessSender(direction, left, right, local_buff);
  } else if (world_rank == receiver) {
    ProcessReceiver(direction, left, right, local_buff);
  } else if (IsOnPath(world_rank, sender, receiver, direction)) {
    ProcessIntermediate(direction, left, right, local_buff);
  }

  int data_size = (world_rank == receiver) ? static_cast<int>(local_buff.size()) : 0;
  MPI_Bcast(&data_size, 1, MPI_INT, receiver, MPI_COMM_WORLD);

  if (world_rank != receiver) {
    local_buff.resize(data_size);
  }
  MPI_Bcast(local_buff.data(), data_size, MPI_INT, receiver, MPI_COMM_WORLD);

  GetOutput() = local_buff;

  return true;
}

int TabalaevALinearTopologyMPI::GetLeft(int rank) {
  return rank == 0 ? MPI_PROC_NULL : rank - 1;
}

int TabalaevALinearTopologyMPI::GetRight(int rank, int size) {
  return rank == (size - 1) ? MPI_PROC_NULL : rank + 1;
}

int TabalaevALinearTopologyMPI::GetDirection(int sender, int receiver) {
  return sender < receiver ? 1 : -1;
}

bool TabalaevALinearTopologyMPI::IsOnPath(int rank, int sender, int receiver, int direction) {
  if (direction == 1) {
    return rank > sender && rank < receiver;
  }
  return rank < sender && rank > receiver;
}

void TabalaevALinearTopologyMPI::ProcessSender(int direction, int left, int right, std::vector<int> &local_buff) {
  local_buff = std::get<2>(GetInput());

  int size = static_cast<int>(local_buff.size());
  int to = (direction == 1 ? right : left);

  MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
}

void TabalaevALinearTopologyMPI::ProcessIntermediate(int direction, int left, int right, std::vector<int> &local_buff) {
  int from = (direction == 1 ? left : right);
  int to = (direction == 1 ? right : left);

  int size = 0;
  MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  local_buff.resize(size);
  MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
  MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
}

void TabalaevALinearTopologyMPI::ProcessReceiver(int direction, int left, int right, std::vector<int> &local_buff) {
  int from = (direction == 1 ? left : right);

  int size = 0;
  MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  local_buff.resize(size);
  MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```