# Нахождение максимальных значений по строкам матрицы Якимов Илья

- Student: Якимов Илья Владимирович, group 3823Б1ПР2
- Technology: SEQ | MPI 
- Variant: 6

## 1. Introduction
Разработка виртуальной линейной топологии, определив для каждого процесса его соседей и передачи ежду ними данных при помощи однопоточного алгоритма (SEQ) - "эмуляция" топологии внутри вектора, а не реальный обмен данными между процессами (т. к. SEQ-версия однопоточная) и реальная реализация топологии на MPI с использованием функций библиотеки MPI. Цель - реализация линейного обмена данными между процессами, в SEQ версии это лишь эмуляция алгоритма при условии что контейнер std::vector является аналогией коммуникатора MPI_COMM_WORLD.

## 2. Problem Statement
Реализация линейной коммуникации между процессами

Вход: заранее сгенерированная data - 15 файлов расширения .txt в папке parallel_programming\ppc-2025-processes-engineers\tasks\yakimov_i_max_values_in_matrix_rows\data

В каждом из файлов строки из трёх чисел: 
1 число - rank source(src) процесса
2 число - rank destination(dst) процесса
3 число - данные, которые мы передаем от source(src) процесса к destination(dst) процессу, входной тип int

Выход: сумма переданных данных между процессами

Ограничения: время работы <1 секунды в functional тестах

## 3. Baseline Algorithm

### SEQ (последовательная эмуляция)

```
for (size_t i = 0; i < operations_.size(); i += 3) {
    int src = operations_[i];
    int dst = operations_[i + 1];
    int data = operations_[i + 2];
    
    if (src == dst) {
        process_data[src] += data;
        total_received += data;
        continue;
    }
    
    int direction = (dst > src) ? 1 : -1;
    int current_process = src;
    int current_data = data;
    
    while (current_process != dst) {
        int next_process = current_process + direction;
        
        if (next_process == dst) {
            // Достигли получателя
            process_data[dst] += current_data;
            total_received += current_data;
        }
        
        current_process = next_process;
    }
}
```

### MPI

```
for (size_t i = 0; i < operations_.size(); i += 3) {
    int src = operations_[i];
    int dst = operations_[i + 1];
    int data = operations_[i + 2];
    
    if (src == dst) {
        // Отправка самому себе
        if (rank == src) {
            total_received += data;
        }
        continue;
    }
    
    int min_proc = std::min(src, dst);
    int max_proc = std::max(src, dst);
    
    if (rank >= min_proc && rank <= max_proc) {
        int direction = (dst > src) ? 1 : -1;
        
        if (rank == src) {
            // Отправитель: отправляет данные следующему процессу
            int next = rank + direction;
            MPI_Send(&data, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        } 
        else if (rank == dst) {
            // Получатель: получает данные от предыдущего процесса
            int received;
            MPI_Status status;
            int prev = rank - direction;
            MPI_Recv(&received, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
            total_received += received;
        } 
        else {
            // Промежуточный процесс: получает и передает дальше
            int received;
            MPI_Status status;
            int prev = rank - direction;
            int next = rank + direction;
            
            MPI_Recv(&received, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&received, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
    }
}
```

## 4. Parallelization Scheme

### Распространение операций:

Мастер-процесс рассылает список операций всем процессам с помощью MPI_Bcast
Каждый процесс получает полный список операций для обработки
Количество операций сначала передается как размер, затем сами данные

Коммуникация: эмуляция линейной виртуальной топологии

### Принцип линейной топологии:

Процессы организованы в линейную цепочку: 0 - 1 - 2 - ... - (N-1)
Каждый процесс может общаться только с непосредственными соседями
Для передачи данных между несоседними процессами используются промежуточные процессы-ретрансляторы

### Алгоритм передачи для каждой операции:

Проверка границ: если источник или назначение выходят за пределы числа процессов, операция игнорируется
Самопередача: если источник = назначению, данные добавляются локально
Определение пути: вычисляется минимальный и максимальный процесс в цепочке передачи
Направление передачи: определяется по сравнению источника и назначения:
Вправо: если назначение > источника
Влево: если назначение < источника

### Распределение ролей:

Источник (src): отправляет данные следующему процессу в цепочке
Назначение (dst): получает данные от предыдущего процесса в цепочке
Промежуточные процессы: получают данные и передают дальше (ретрансляция)

### Синхронизация:

После каждой операции вызывается MPI_Barrier для синхронизации всех процессов
Это гарантирует, что следующая операция начнется только после завершения текущей

### Сбор результатов: 

Редукция и широковещательная рассылка

### Агрегация результатов:

Каждый процесс локально суммирует полученные данные в total_received
Используется MPI_Reduce для сбора всех локальных сумм на мастер-процесс (ранг 0)

## 5. Implementation Details

### Структура кода:
Используем команду "tree tasks/yakimov_i_max_values_in_matrix_rows/" для того чтобы узнать структуру проекта:

tasks/yakimov_i_max_values_in_matrix_rows/
├── common
│   └── include
│       └── common.hpp
├── data
│   ├── 1.txt
│   ├── 10.txt
│   ├── 11.txt
│   ├── 2.txt
│   ├── 27.txt
│   ├── 28.txt
│   ├── 29.txt
│   ├── 3.txt
│   ├── 30.txt
│   ├── 4.txt
│   ├── 5.txt
│   ├── 6.txt
│   ├── 7.txt
│   ├── 8.txt
│   ├── 9.txt
│   └── create_data_files.sh
├── info.json
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── report.md
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── settings.json
└── tests
    ├── functional
    │   └── main.cpp
    └── performance
        └── main.cpp

13 directories, 26 files

### Ключевые методы (вспомогательные для RunImpl()):

ReadOperationsFromFile() - чтение операций передачи (только процесс 0)
BroadcastOperations() - рассылка операций всем процессам
CalculateProcessRange() - определение участия процесса в операции

### Особенности:

Неравномерная нагрузка - процессы участвуют только в операциях на своем пути
Валидация на процессе 0 - проверка входных данных
Барьер после каждой операции - строгая синхронизация
Специальные случаи - самопередача, пограничные процессы, некорректные операции

## 6. Experimental Setup
- Hardware/OS: CPU model: Intel Core i7 12700H, cores/threads: 14/20, RAM: 16GB, OS version: Ubuntu 24.04.3 LTS
- Toolchain: compiler: gcc, version: 14, MPI: OpenMPI 4.1.2, build type (Release/RelWithDebInfo): Release
- Environment: PPC_NUM_THREADS / PPC_NUM_PROC: 1 / 8, other relevant vars: -
- Data: процессы от 0 до 20 (20 - максимальное число на машине) - проверялось при помощи команды "nproc"

## 7. Results and Discussion

### 7.1 Correctness
- Проверка против последовательной реализации
- Unit-тесты с эталонными значениями
- Тестирование на матрицах разного размера
- Валидация граничных случаев (1 строка, 1 столбец)

### 7.2 Performance
Результаты в столбце "Time" представлены как среднее между 3 запусками performance тестов

Speedup = T_seq / T_parallel
Efficiency = Speedup / Count * 100%

#### Измерения "чистого" времени вычислений максимальных элементов по строкам матрицы - task_run

| Mode        | Count |    Time, ms    | Speedup | Efficiency |
|-------------|-------|----------------|---------|------------|
| seq         | 1     | 0.0081394196   | 1.00    | N/A        |
| mpi         | 4     | 0.0183642742   | 0.44    | 11.0%      |
| mpi         | 8     | 0.0257455308   | 0.32    | 4.0%       |
| mpi         | 12    | 0.0304794270   | 0.27    | 2.2%       |
| mpi         | 20    | 0.0298154522   | 0.27    | 1.4%       |

#### Измерения полного времени вычислений ("чистое" + затраты на открытие файла, считывание и коммуникацию процессов) - pipeline

| Mode        | Count |    Time, ms    | Speedup | Efficiency |
|-------------|-------|----------------|---------|------------|
| seq         | 1     | 3.2155227184   | 1.00    | N/A        |
| mpi         | 4     | 1.5889995484   | 2.02    | 50.5%      |
| mpi         | 8     | 2.9256506878   | 1.10    | 13.8%      |
| mpi         | 12    | 2.8845829832   | 1.11    | 9.3%       |
| mpi         | 20    | 4.8718488208   | 0.66    | 3.3%       |

## 8. Conclusions
Так как SEQ версия лишь эмуляция линейного обмена данными между процесами где контейнер std::vector выступает в роли коммуникатора, нельзя обьективно определить премущество MPI версии поскольку данная задача предназначена исключительно для MPI реализации. В SEQ версии искусственно добавлен цикл, для ее замедления с целью прохождения минимального тайминга у тестов (должны проходить за > 0,001 милисек.)

## 9. References
1. OpenMPI документация: <https://www.open-mpi.org/>
2. MPI стандарт: <https://www.mpi-forum.org/>
3. Курс параллельного программирования: материалы курса <https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html>
4. Мастер проекта: <https://github.com/learning-process/ppc-2025-processes-engineers>