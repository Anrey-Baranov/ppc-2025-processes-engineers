# Линейная фильтрация изображений (вертикальное разбиение). Ядро Гаусса 3x3
  - Студент: Кондрашова Виктория Андреевна, group 3823Б1ПР1
  - Технологии: SEQ | MPI
  - Вариант: 27

## 1. Введение
Фильтр Гаусса широко применяется для размытия изображений, подавления шума, сглаживания и предварительной обработки перед применением других алгоритмов (детекция границ, сегментация и др.). В рамках данного исследования проводится сравнительный анализ производительности последовательной и параллельной MPI-реализаций алгоритма гауссовой фильтрации с использованием вертикального разбиения изображения.

## 2. Постановка задачи
**Цель работы:** Реализовать последовательную и MPI-параллельную версии алгоритма линейной фильтрации изображений с ядром Гаусса 3x3, провести их сравнение и анализ эффективности.

**Определение задачи:** Для изображения размера width × height с количеством каналов channels необходимо применить свёрточный фильтр с ядром Гаусса 3x3.

**Ограничения:**
 - Корректность вычислений должна сохраняться при любых значениях пикселей.
 - Минимальный размер изображения: 3×3 пикселя (для корректного применения ядра 3×3).
 - Результаты SEQ и MPI версий должны полностью совпадать.
 - Входные данные: структура ImageData, содержащая вектор пикселей и размеры изображения.
 - Выходные данные: структура ImageData с отфильтрованным изображением тех же размеров.
 - Граничные случаи: Для обработки пикселей на краях изображения применяется метод зажатия координат (clamp). При выходе за границы изображения координаты ограничиваются допустимым диапазоном. Данный подход похож на «растягивание» крайних пикселей за пределы изображения. Например, для левого верхнего угла (0,0) при обращении к несуществующему пикселю (-1,-1) будет использовано значение пикселя (0,0). При вертикальном разбиении изображения между MPI-процессами каждому процессу передаётся расширенная область данных, включающая по одному дополнительному столбцу с каждой стороны.

## 3. Алгоритм(Последовательная версия)
**Шаги алгоритма:**

 - Валидация: Проверка минимального размера изображения (width ≥ 3, height ≥ 3), корректности количества каналов (1 ≤ channels ≤ 4), проверка соответствия размера вектора пикселей (size = width × height × channels)
 - Инициализация: Создание выходного изображения с теми же размерами, выделение памяти под результирующие пиксели

 - Применение фильтра Гаусса:

  -- Для каждой строки y от 0 до height-1:
  -- Для каждого столбца x от 0 до width-1:
  -- Для каждого канала c от 0 до channels-1:
  -- Вычисление взвешенной суммы соседних пикселей с использованием ядра Гаусса
  -- Обработка граничных условий методом clamp (повторение крайних пикселей)
  -- Нормализация результата делением на 16
  -- Запись результата в выходное изображение

## 4. Схема распараллеливания
Этапы выполнения
### 1. Валидация (ValidationImpl)
Процесс 0:**

Проверяет корректность входных данных

**Остальные процессы:**

Возвращают true

### 2. Предобработка (PreProcessingImpl)
**Процесс 0:**

-- Извлекает размеры изображения из входных данных
-- Инициализирует выходную структуру ImageData
-- Выделяет память под результат

### 3. Основное вычисление (RunImpl)
### 3.1. Рассылка размеров изображения (BroadcastImageDimensions)
### 3.2. Распределение столбцов между процессами (CalculateColumnDistribution)
### 3.3. Распределение данных изображения (DistributeImageData)
### 3.4. Локальное применение фильтра (ApplyGaussFilterToLocalData)
### 3.5. Сбор результатов (GatherResults)
### 3.6. Рассылка результата всем процессам (BroadcastResultToAllProcesses)

### 4. Постобработка (PostProcessingImpl)
**Все процессы:**

Сразу возвращают true  (результат уже готов после RunImpl)

## 5.Детали реализации
### Структура кода

**Файлы:**
- `common/include/common.hpp` - определение типов данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - последовательная реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - параллельная реализация
- `tests/functional/main.cpp` - функциональные тесты
- `tests/performance/main.cpp` - тесты производительности

**Классы:**
- `KondrashovaVGaussFilterVerticalSplitSEQ` - последовательная версия (SEQ)
- `KondrashovaVGaussFilterVerticalSplitMPI` - параллельная версия (MPI)

**Основные методы (одинаковы для обеих реализаций):**
- `ValidationImpl()` - проверка корректности входа
- `PreProcessingImpl()` - подготовка данных
- `RunImpl()` - основная логика (последовательная или MPI).
- `PostProcessingImpl()` - проверка корректности выхода.


## 6. Экспериментальная установка
- Hardware/OS: CPU - AMD Ryzen 5 5600H, 6 ядер/12 потоков; RAM - 16 Gb; ОС - Windows 10 
- Toolchain: g++ 11.4.0 , build type: Release  
- Environment: PPC_NUM_PROC
- Data: тестовые данные задаются вручную.

## 7. Результаты и обсуждения

### 7.1 Корректность

**Корректность проверена через:**

* Модульные тесты для различных размеров изображений

* Сравнение между последовательными и MPI результатами

* Проверка граничных условий и валидации

* Все функциональные тесты успешно пройдены

### 7.2 Производительность
**Характеристики задачи:**
- Размер: 3840 × 2160 изображение в 4К

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 4.413   | 1.00    | N/A        |
| mpi         | 2     | 3.418   | 1.3     | 65%        |
| mpi         | 4     | 2.991   | 1.48    | 37%        |
| mpi         | 8     | 2.798   | 1.58    | 19%        |

## 8. Вывод
ВВ ходе выполнения работы была создана и протестирована программная реализация задачи линейной фильтрации изображений с использованием ядра Гаусса 3×3. Разработка включала два подхода: классический последовательный алгоритм и параллельную версию с применением технологии MPI.

Разработанные алгоритмы корректно решают поставленную задачу, что подтверждается успешным прохождением всех функциональных тестов. Результаты SEQ и MPI версий полностью совпадают для всех тестовых изображений различных размеров и форматов.

Из полученных результатов видно, что MPI-версия демонстрирует преимущество за счёт распределения вычислительной нагрузки между процессами.
## 9. Источники
1. Список лекций по курсу "Параллельное программирование". (Сысоев А.В. ННГУ 2025 г.)
2. Список практических занятий по курсу "Пареллельное программирование". (Оболенский А.А, ННГУ 2025 г.)
3. Документация по курсу: "Параллельное программирование": <https://learning-process.github.io/parallel_programming_course/ru/index.html> (Оболенский А.А, Нестеров А.Ю)

## 10. Приложение

### MPI-реализация(ключевой алгоритм)
```cpp
bool KondrashovaVGaussFilterVerticalSplitMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int width = 0;
  int height = 0;
  int channels = 0;
  BroadcastImageDimensions(width, height, channels);

  std::vector<int> col_counts;
  std::vector<int> col_offsets;
  CalculateColumnDistribution(width, size, col_counts, col_offsets);

  int local_start_col = col_offsets[rank];
  int local_cols = col_counts[rank];

  int extended_start = std::max(0, local_start_col - 1);
  int extended_end = std::min(width, local_start_col + local_cols + 1);
  int extended_cols = extended_end - extended_start;
  int offset_in_extended = local_start_col - extended_start;

  std::vector<uint8_t> local_data;
  DistributeImageData(rank, size, width, height, channels, col_counts, col_offsets, local_data, extended_cols);

  std::vector<uint8_t> local_result;
  ApplyGaussFilterToLocalData(local_data, local_result, extended_cols, local_cols, height, channels,
                              offset_in_extended);

  GatherResults(rank, size, width, height, channels, col_counts, col_offsets, local_start_col, local_cols,
                local_result);

  BroadcastResultToAllProcesses(width, height, channels);

  return true;
}

void KondrashovaVGaussFilterVerticalSplitMPI::ApplyGaussFilterToLocalData(const std::vector<uint8_t> &local_data,
                                                                          std::vector<uint8_t> &local_result,
                                                                          int extended_cols, int local_cols, int height,
                                                                          int channels, int offset_in_extended) {
  local_result.resize(static_cast<size_t>(local_cols) * height * channels);

  for (int row = 0; row < height; ++row) {
    for (int lx = 0; lx < local_cols; ++lx) {
      int col = offset_in_extended + lx;
      for (int ch = 0; ch < channels; ++ch) {
        int result_idx = (((row * local_cols) + lx) * channels) + ch;
        local_result[result_idx] = ApplyGaussToLocalPixel(local_data, extended_cols, height, channels, col, row, ch);
      }
    }
  }
}
```
