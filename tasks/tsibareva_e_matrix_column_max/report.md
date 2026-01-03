# Нахождение максимальных значений по столбцам матрицы

- Студент: Цибарева Екатерина Алексеевна, группа 3823Б1ПР1
- Технология: SEQ | MPI
- Вариант: 16

## 1. Введение
Задача поиска максимумов по столбцам матрицы критически важна для анализа данных в машинном обучении, финансовом моделировании и научных вычислениях. При работе с матрицами большого размера последовательные алгоритмы показывают не удовлетворительные результаты, критически сказывающиеся на общей производительности вычислений.

Целью данной работы является реализация и сравнительный анализ последовательного и параллельного алгоритмов для нахождения максимальных значений и их позиций в каждом столбце матрицы произвольной размерности.

## 2. Постановка задачи
**Описание задачи**

Для каждого столбца заданной матрицы найти значение максимального элемента.\
Входной тип данных: вектор векторов значений типа int.\
Выходной тип данных: вектор значений типа int.

**Ограничения**
- матрица может состоять из повторяющихся значений либо содержать повторяющиеся значения;
- результирующие вектора последовательной и параллельной реализаций алгоритма не должны различаться;
- для реализации параллельного алгоритма должен быть использован MPI;
- матрица может быть квадратной или прямоугольной;
- матрица может быть подана на вход алгоритма в не корректном формате, а именно - иметь строки различной длины, быть пустой или содержать только пустые столбцы.

## 3. Описание базового алгоритма
**Основная идея**

Базовый, или последовательный, алгоритм выполняет поиск максимальных значений для каждого столбца матрицы путем последовательного обхода всех элементов. Необходимо для каждого столбца независимо найти наибольший элемент, просматривая все строки этого столбца.

Конвейер выполнения как последовательного, так и параллельного алгоритма предполагает обязательную проверку на не пустоту матрицы и не пустоту её столбцов. Также происходит проверка на равенство длин строк матрицы. 
``` cpp
if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    return true;
}
```

**Шаги базового (последовательного) алгоритма обработки**: 

1. Инициализация матрицы
``` cpp
const auto &matrix = GetInput();
```
2. Обязательная проверка на пустоту матриц, штатная ситуация возращаемого значения для пустых матриц и матриц, в которых строки имеют разную длину (непрямоугольных) - возвращение пустого вектора
3. Инициализация выходного вектора
``` cpp
auto &column_maxs = GetOutput();
```
4. Вычисление количества столбцов
``` cpp
size_t cols_count = matrix[0].size();
```
5. Алгоритм предполагает последовательный обход всех элементов с квадратичной (m*n, где m и n - количество строк и столбцов соответственно) сложностью. В качестве первоначально присваиваемого максимального значения выбран элемент первой строки в цикле.

Частичный код базового (последовательного) алгоритма поиска максимумов матрицы по столбцам можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  auto &column_maxs = GetOutput();
  size_t cols_count = matrix[0].size();

  for (size_t col = 0; col < cols_count; ++col) {
    int max_value = matrix[0][col];
    for (size_t row = 1; row < matrix.size(); ++row) {
      max_value = std::max(matrix[row][col], max_value);
    }
    column_maxs[col] = max_value;
  }

  return true;
}
```

## 4. Схема распараллеливания
**Модель распределения**

Предположено циклическое распределение работы между процессами таким образом, чтобы каждый процесс обрабатывал порядковые номера столбцов с шагом, равным world_size. Иначе говоря, процесс с рангом rank должен обрабатывать столбцы с номерами rank, rank + world_size, rank + 2*world_size и так далее.

**Роли процессов**

Процесс 0 выступает в роли координатора:
- выполняет локальные вычисления;
- собирает частичные результаты с процессов рангом 1 и выше;
- формирует результирующий вектор в соответствии с моделью рапределения столбцов между процессами;
- рассылает финальный результат.

Процессы рангом 1 и выше:
- выполняют локальные вычисления максимумов для назначенных им столбцов;
- передают результаты локальных вычислений процессу с рангом 0.

**Коммуникационная схема**

**Шаги параллельного алгоритма обработки**: 

1. Вычисление процессом локально максимумов по назначенным ему столбцам.
      ``` cpp
      for (auto col = static_cast<size_t>(world_rank); col < cols_count; col += static_cast<size_t>(world_size)) {
       int maxum_value = matrix[0][col];
       for (size_t row = 1; row < rows_count; ++row) {
         maxum_value = std::max(matrix[row][col], maxum_value);
       }
       local_maxs.push_back(maxum_value);
      }
      ```
2. Сбор и обобщение результата:
    - процессы с рангом 1 и выше направляют результаты вычислений процессу 0 с помощью MPI_Send;
    ``` cpp
      if (!local_maxs.empty()) {
        MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
    ```
    - процесс с рангом 0 принимает результаты через MPI_Recv и формирует результирующий вектор;
    ``` cpp
      std::vector<int> proc_maxs(static_cast<size_t>(proc_pass));
      MPI_Recv(proc_maxs.data(), proc_pass, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      size_t proc_idx = 0;
      for (size_t col = proc; col < cols_count; col += world_size) {
        final_result_[col] = proc_maxs[proc_idx++];
      }
    ```
3. Рассылка финального результата процессом 0 между остальными процессами при помощи MPI_BCast (что гарантирует идентичность выходных данных на всех процессах и позволяет более корректно и просто проследить успешное прохождение всех запущенных процессов).
      ``` cpp
      MPI_Bcast(GetOutput().data(), static_cast<int>(cols_count), MPI_INT, 0, MPI_COMM_WORLD);
      ```

**Особенности модели распределения и коммуникационной схемы**:
- циклическое распределение столбцов позволяет не только балансировать нагрузку, равномерно распределяя столбцы между процессами, но и вводить логическое условие, универсально ограничивающее обработку матрицы меньшего количества столбцов, чем запущенных процессов; в таком случае происходит не выделение процессу с рангом, большим или равным количеству столбцов матрицы, никаких данных для обработки;
- минимизация передаваемых данных в пользу реализации алгоритма, пересылающего только значения локальных максимумов с циклическим восстановлением последовательности результирующего вектора, против пересылки и значений, и номеров максимумов в векторе;
- обработка основных граничных случаев для матриц, содержащих как больше, так и меньше столбцов, чем запущенных для обработки процессов.

Частичный код реализации алгоритма параллельной обработки можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxMPI::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &matrix = GetInput();
  size_t rows_count = matrix.size();
  size_t cols_count = matrix[0].size();

  std::vector<int> local_maxs;

  for (auto col = static_cast<size_t>(world_rank); col < cols_count; col += static_cast<size_t>(world_size)) {
    int maxum_value = matrix[0][col];
    for (size_t row = 1; row < rows_count; ++row) {
      maxum_value = std::max(matrix[row][col], maxum_value);
    }
    local_maxs.push_back(maxum_value);
  }

  if (world_rank == 0) {
    CollectResultsFromAllProcesses(local_maxs, world_size, cols_count);
  } else {
    if (!local_maxs.empty()) {
      MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}
```
```cpp
void TsibarevaEMatrixColumnMaxMPI::CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size,
                                                                  size_t cols_count) {
  final_result_.resize(cols_count);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t idx = 0;
  for (size_t col = 0; col < cols_count && idx < local_maxs.size(); col += world_size) {
    final_result_[col] = local_maxs[idx++];
  }

  for (int proc = 1; proc < world_size; proc++) {
    int proc_pass = 0;

    for (size_t col = proc; col < cols_count; col += world_size) {
      proc_pass++;
    }

    if (proc_pass <= 0) {
      continue;
    }

    std::vector<int> proc_maxs(static_cast<size_t>(proc_pass));
    MPI_Recv(proc_maxs.data(), proc_pass, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    size_t proc_idx = 0;
    for (size_t col = proc; col < cols_count; col += world_size) {
      final_result_[col] = proc_maxs[proc_idx++];
    }
  }
}
```

## 5. Детали реализации

**Структура проекта**
``` text
tsibareva_e_matrix_column_max
        │   info.json
        │   report.md
        │   settings.json
        ├───common
        │   └───include
        │           common.hpp
        ├───data
        │       pic.jpg
        ├───mpi
        │   ├───include
        │   │       ops_mpi.hpp
        │   │
        │   └───src
        │           ops_mpi.cpp
        ├───seq
        │   ├───include
        │   │       ops_seq.hpp
        │   │
        │   └───src
        │           ops_seq.cpp
        └───tests
            │   .clang-tidy
            │
            ├───functional
            │       main.cpp
            └───performance
                    main.cpp
```

Проект реализации исходных алгоритмов, функционального тестирования и тестирования производительности имеет сложную структуру. 


**Файлы реализаций**

1. Последовательная реализация (seq):

- ops_seq.hpp - объявление класса TsibarevaEMatrixColumnMaxSEQ, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl());

- ops_seq.cpp - реализация перегруженных методов:
    - RunImpl()            - базовый (последовательный) алгоритм поиска максимумов;
    - PreProcessingImpl()  - инициализирован выходной вектор, проверены случаи не корректного задания матрицы (пустота, неравенство длин строк);
    - ValidationImpl()     - не предположено отдельной логики;
    - PostProcessingImpl() - не предположено отдельной логики.

2. MPI реализация (mpi):

- ops_mpi.hpp - объявление класса TsibarevaEMatrixColumnMaxMPI, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl()), объявлен вектор final_result_;

- ops_mpi.cpp - реализация методов:
    - ValidationImpl()     - не предположено отдельной логики;
    - PreProcessingImpl()  - инициализирован выходной вектор и вектор final_result_, проверены случаи не корректного задания матрицы (пустота, неравенство длин строк);
    - RunImpl()            - параллельный алгоритм поиска максимумов, распределяющий содержательные действия между процессами;
    - PostProcessingImpl() - рассылка финального результата между процессами в конце выполнения обработки;
    - CollectResultsFromAllProcesses(const std::vector<int> &local_maxs, int world_size, size_t cols_count) - метод, используемый только процессом 0, осуществляющим координирование вычислений между процессами; предполагает сбор частичных результатов и формирование результирующего вектора;
    
Необходимо отметить, что для координирования действий между функциями параллельного алгоритма была введена глобальная переменная std::vector<int> final_result_, временно выполняющая хранение результирующих данных, собранных на нулевом процессе.


**Тестирование и формирование входных данных**

3. Общие компоненты (common):
- common.hpp содержит:
    - объявление входного (InType), выходного (OutType) типа, типа базовой задачи (BaseTask) и тестовых классов (TestTask); необходимо отметить, что входные данные представлены в формате вектора векторов, выходные - вектора максимумов, а тестовый класс состоит из содержательного типа матрицы и строки её краткого описания; 
    ``` cpp
    using InType = std::vector<std::vector<int>>;
    using OutType = std::vector<int>;
    using TestType = std::tuple<MatrixType, std::string>;
    ``` 
    - перечисление типов матриц, доступных для генерации (enum MatrixType), более подробно описанных в пункте 7.1 "Корректность";
    - функции формирования матриц 20 типов в соответствии с перечислением;
    - функцию задания матрицы GenerateMatrixFunc, в качестве аргументов принимающую тип генерируемой матрицы, и принимающей решение о вызове соответствующей функции генерации матрицы;
    - функцию GenerateExpectedOutput, задающую эталонный вектор выходных значений максимумов по столбцам для каждого типа матрицы.

Необходимо отметить, что функции формирования матриц были реализованы таким образом, что задают заранее известные наборы данных без генерации произвольных матриц. Выбор был сделан в пользу предопределенных сформированных матриц и векторов с целью наиболее точного сравнения результатов с известным эталонным значением. Например, генерация возрастающей квадратной матрицы 8 на 8 и её эталонного вектора представлена ниже.
``` cpp
inline std::vector<std::vector<int>> GenerateAscendingMatrix() {
  return {
    {1,  2,  3,  4,  5,  6,  7,  8},
    {9, 10, 11, 12, 13, 14, 15, 16},
    {17, 18, 19, 20, 21, 22, 23, 24},
    {25, 26, 27, 28, 29, 30, 31, 32},
    {33, 34, 35, 36, 37, 38, 39, 40},
    {41, 42, 43, 44, 45, 46, 47, 48},
    {49, 50, 51, 52, 53, 54, 55, 56},
    {57, 58, 59, 60, 61, 62, 63, 64}};
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}
```

4. Функциональные тестовые файлы (tests/funtional):
- main.cpp - содержит:
    - объявление класса TsibarevaERunFuncTestsProcesses;
    - переопределение функции PrintTestParam таким образом, что печать тестов содержит краткое описание её типа, включающее размерность;
    - переопределение функции SetUp таким образом, что исходная матрица может быть получена как GenerateMatrixFunc(<тип матрицы 1 из 20 перечисленных>), а ожидаемые данные - как GenerateExpectedOutput(<тот же тип матрицы>);
    ``` cpp
    MatrixType matrix_type = std::get<0>(params);
    input_data_            = GenerateMatrixFunc(matrix_type);
    expected_output_       = GenerateExpectedOutput(matrix_type);
    ```
    - инициализацию и заполнение массива аргументов 20 тестовыми случаями, выполнимыми как для последовательной, так и для параллельной реализации алгоритма; тестовые аргументы покрывают такие случаи, как единичная матрица, матрица, состоящая из одной строки/столбца, квадратная матрица, прямоугольная матрица, у которой количество столбцов/строк превышает количество строк/столбцов соответственно; тестовые аргументы покрывают ситуации, в которых на вход алгоритму передана пустая матрица или матрица, состоящая из строк различной длины.

5. Тестовые файлы производительности (tests/performance):
- main.cpp - содержит:
    - объявление класса TsibarevaERunPerfTestProcesses;
    - переопределение функции SetUp таким образом, что исходная матрица и эталонный вектор (опциональный в случае нагрузочного тестирования) максимумов по столбцам в ней формируются на фиксированное количество элементов, размером 10000 на 10000 элементов для запуска Github Actions и 6000 на 6000 - для локального запуска. 

Необходимо отметить, что в случае генерации объемных матриц для тестирования производительности сформирована квадратная матрица с максимумом в середине столбца. Кроме того, ни одна реализация формирования тестовых данных не использует случайную генерацию чисел ни в каком виде, что достаточно для покрытия основных сценариев работы и предполагает стабильное выполнение алгоритма при многократных запусках.

## 6. Экспериментальные результаты

**Аппаратное обеспечение и характеристики ОС локального запуска**:
- Модель процессора: AMD Ryzen 7 5700U (1.80 GHz)
- Архитектура: x86-64
- Ядра: 8 ядер
- Оперативная память: 8 GB
- Операционная система: Windows 10 Home (базовая) / Ubuntu 24.04.3 LTS (сборочная)
- Подсистема: WSL2 (Windows Subsystem for Linux)

**Инструменты**:
- Компилятор: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- MPI реализация: Open MPI 4.1.6
- Тип сборки: Release 

**Настройки окружения**:
- PPC_NUM_PROC: 2 (количество MPI процессов)
- PPC_NUM_THREADS: 4 (доступные потоки)
- Количество доступных процессов PPC_NUM_PROC также может быть задано ключом ``` mpirun -n M ``` при запуске тестов, где M - требуемое количество процессов.

## 7. Результаты и обсуждение

### 7.1 Корректность
Для проверки корректности выполнения алгоритмами вычисления максимальных элементов в столбцах были использованы предварительно подготовленные матрицы и эталонные вектора максимумов 20 типов. Каждый тип был прописан в enum MatrixType и предполагает как покрытие различных размерностей матриц, так и различных типов их формирования. Так, тестовые матрицы могут быть следующих видов:
| Тип                 | Краткое описание                |
|---------------------|---------------------------------|
| kSingleConstant     | 1x1 константная                 |
| kSingleRow          | 1x10 одна строка                |
| kSingleCol          | 3x1 один столбец                |
| kAllZeros           | 5x5 все нули                    |
| kConstant           | 5x5 константная                 |
| kMaxFirst           | 6x4 максимум в первой строке    |
| kMaxLast            | 6x4 максимум в последней строке |
| kMaxMiddle          | 6x4 максимум в середине         |
| kAscending          | 8x8 возрастающая                |
| kDescending         | 8x8 убывающая                   |
| kDiagonalDominant   | 8x8 диагонально доминантная     |
| kSparse             | 8x8 разреженная                 |
| kNegative           | 8x8 отрицательная               |
| kSquareSmall        | 2x2 маленькая квадратная        |
| kVertical           | 10x4 вертикальная               |
| kHorizontal         | 5x10 горизонтальная             |
| kCheckerboard       | 7x7 шахматная                   |
| kEmpty              | Пустая матрица                  |
| kZeroColumns        | Матрица с пустыми столбцами     |
| kNonRectangular     | Непрямоугольная матрица         |

Был предположен отказ от генерации матриц посредством функций в пользу формирования изначально предопределенных наборов данных для матриц и эталонных векторов. Эталонные вектора, возвращаемые по аргументу типа соответствующего типу формирования матрицы, содержат максимумы этих матриц. 

Несмотря на то, что для функционального тестирования метод задания предопределённых данных корректен, в силу ограниченности максимальных объемов файлов проекта такой способ не может быть применим к генерации данных для тестирования производительности. Исходя из чего, для performance тестирования была предположена генерация матриц в зависимости от количества строк и столбцов (по 6 или 10 тысяч), и подсчет эталонного вектора максимумов в процессе генерации матрицы.

Полный код формирования тестовых случаев функционального тестирования можно видеть в приложении 1.

### 7.2 Производительность
Тесты производительности были проведены локально на 36 миллионах элементов.

### Режим Pipeline
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.2983   | 1.00      | N/A           |
| 1 (MPI)   | 0.2982   | 1.00      | 100%          |
| 2 (MPI)   | 0.1907   | 1.56      | 78.2%         |
| 3 (MPI)   | 0.1573   | 1.90      | 63.2%         |
| 4 (MPI)   | 0.1775   | 1.68      | 42.0%         |

### Режим Task Run  
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.2956   | 1.00      | N/A           |
| 1 (MPI)   | 0.2950   | 1.00      | 100%          |
| 2 (MPI)   | 0.1869   | 1.58      | 79.1%         |
| 3 (MPI)   | 0.1365   | 2.17      | 72.2%         |
| 4 (MPI)   | 0.1701   | 1.74      | 43.5%         |

Необходимо отметить, что ускорение каждого запуска было высчитано относительно времени выполнения последовательной версии на одном процессе, а эффективность определена как ускорение, поделенное на количество запущенных процессов.

Таким образом, параллельная реализация алгоритма в обоих режимах показала ускорение от 1.56 до 2.17 раз относительно последовательной. 
Издержки эффективности на накладные расходы параллельного алгоритма с использованием MPI, которые показали линейный рост с увеличением числа процессов были обусловлены локальным тестированием на относительно небольшом количестве элементов (десятки, но не сотни миллионов). 

## 8. Выводы
Задача реализации алгоритма поиска максимумов в матрице по столбцам была решена последовательным образом и параллельным образом при использовании OpenMPI. Были предложены различные способы формирования исходных данных для функциональных тестов и тестов производительности. А также выполнены замеры времени выполнения последовательного и параллельного алгоритмов с последующим высчитыванием эффективности распараллеливания вычислений.

По результатам замеров, параллельный алгоритм показал ускорение выполнения задачи в ~1,6-2,7 раз. Циклическая схема распределения данных при этом предполагает балансировку нагрузки. 

## 9. Источники

1. **Технологии параллельного программирования MPI и OpenMP** // А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.
2. **Инструменты параллельного программирования в системах с общей памятью: Учебное пособие.** // Корняков К.В., Мееров И.Б., Сиднев А.А., Сысоев А.В., Шишков А.В., - Нижний Новгород: Изд-во Нижегородского госуниверситета, 2010. - 202 с.
3. **Справочник по MPI** // URL: https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference, 2023 (дата обращения: 15.11.2025).
4. **Open MPI: Open Source High Performance Computing** // URL: https://www-lb.open-mpi.org/doc/v4.1, 2025 (дата обращения: 15.11.2025).

### 10. Приложения

## Приложение 1: генерация данных для функциональных тестов
``` cpp
namespace tsibareva_e_matrix_column_max {

enum class MatrixType : std::uint8_t {
  kSingleConstant,    // 1x1 константная
  kSingleRow,         // 1x10 одна строка
  kSingleCol,         // 3x1 один столбец
  kAllZeros,          // 5x5 все нули
  kConstant,          // 5x5 константная
  kMaxFirst,          // 6x4 максимум в первой строке
  kMaxLast,           // 6x4 максимум в последней строке
  kMaxMiddle,         // 6x4 максимум в середине
  kAscending,         // 8x8 возрастающая
  kDescending,        // 8x8 убывающая
  kDiagonalDominant,  // 8x8 диагонально доминантная
  kSparse,            // 8x8 разреженная
  kNegative,          // 8x8 отрицательная
  kSquareSmall,       // 2x2 маленькая квадратная
  kVertical,          // 10x4 вертикальная
  kHorizontal,        // 5x10 горизонтальная
  kCheckerboard,      // 7x7 шахматная
  kEmpty,             // Пустая матрица
  kZeroColumns,       // Матрица с нулевыми столбцами
  kNonRectangular     // Непрямоугольная матрица
};

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::vector<std::vector<int>> GenerateSingleConstantMatrix() {
  return {{30}};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {30};
}

inline std::vector<std::vector<int>> GenerateSingleRowMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
}

inline std::vector<std::vector<int>> GenerateSingleColMatrix() {
  return {{1}, {2}, {3}};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3};
}

inline std::vector<std::vector<int>> GenerateAllZerosMatrix() {
  return {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0, 0};
}

inline std::vector<std::vector<int>> GenerateConstantMatrix() {
  return {{30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}, {30, 30, 30, 30, 30}};
}
inline std::vector<int> GenerateConstantExpected() {
  return {30, 30, 30, 30, 30};
}

inline std::vector<std::vector<int>> GenerateMaxFirstMatrix() {
  return {{1000, 1001, 1002, 1003}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}};
}
inline std::vector<int> GenerateMaxFirstExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxLastMatrix() {
  return {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}, {1000, 1001, 1002, 1003}};
}
inline std::vector<int> GenerateMaxLastExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateMaxMiddleMatrix() {
  return {{1, 2, 3, 4}, {2, 3, 4, 5}, {1000, 1001, 1002, 1003}, {4, 5, 6, 7}, {5, 6, 7, 8}, {6, 7, 8, 9}};
}
inline std::vector<int> GenerateMaxMiddleExpected() {
  return {1000, 1001, 1002, 1003};
}

inline std::vector<std::vector<int>> GenerateAscendingMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8},         {9, 10, 11, 12, 13, 14, 15, 16},  {17, 18, 19, 20, 21, 22, 23, 24},
          {25, 26, 27, 28, 29, 30, 31, 32}, {33, 34, 35, 36, 37, 38, 39, 40}, {41, 42, 43, 44, 45, 46, 47, 48},
          {49, 50, 51, 52, 53, 54, 55, 56}, {57, 58, 59, 60, 61, 62, 63, 64}};
}
inline std::vector<int> GenerateAscendingExpected() {
  return {57, 58, 59, 60, 61, 62, 63, 64};
}

inline std::vector<std::vector<int>> GenerateDescendingMatrix() {
  return {{64, 63, 62, 61, 60, 59, 58, 57}, {56, 55, 54, 53, 52, 51, 50, 49}, {48, 47, 46, 45, 44, 43, 42, 41},
          {40, 39, 38, 37, 36, 35, 34, 33}, {32, 31, 30, 29, 28, 27, 26, 25}, {24, 23, 22, 21, 20, 19, 18, 17},
          {16, 15, 14, 13, 12, 11, 10, 9},  {8, 7, 6, 5, 4, 3, 2, 1}};
}
inline std::vector<int> GenerateDescendingExpected() {
  return {64, 63, 62, 61, 60, 59, 58, 57};
}

inline std::vector<std::vector<int>> GenerateDiagonalDominantMatrix() {
  return {{1000, 1, 2, 3, 4, 5, 6, 7},    {1, 1100, 3, 4, 5, 6, 7, 8},    {2, 3, 1200, 5, 6, 7, 8, 9},
          {3, 4, 5, 1300, 7, 8, 9, 10},   {4, 5, 6, 7, 1400, 9, 10, 11},  {5, 6, 7, 8, 9, 1500, 11, 12},
          {6, 7, 8, 9, 10, 11, 1600, 14}, {7, 8, 9, 10, 11, 12, 14, 1700}};
}
inline std::vector<int> GenerateDiagonalDominantExpected() {
  return {1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700};
}

inline std::vector<std::vector<int>> GenerateSparseMatrix() {
  return {{8, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 61, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 72, 0},
          {0, 0, 0, 0, 0, 0, 0, 0}, {0, 9, 0, 0, 0, 0, 0, 0},  {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
}
inline std::vector<int> GenerateSparseExpected() {
  return {8, 9, 0, 61, 0, 0, 72, 0};
}

inline std::vector<std::vector<int>> GenerateNegativeMatrix() {
  return {{-30, -31, -32, -33, -34, -35, -36, -37}, {-40, -41, -42, -43, -44, -45, -46, -47},
          {-50, -51, -52, -53, -54, -55, -56, -57}, {-60, -61, -62, -63, -64, -65, -66, -67},
          {-70, -71, -72, -73, -74, -75, -76, -77}, {-80, -81, -82, -83, -84, -85, -86, -87},
          {-90, -91, -92, -93, -94, -95, -96, -97}, {-100, -101, -102, -103, -104, -105, -106, -107}};
}
inline std::vector<int> GenerateNegativeExpected() {
  return {-30, -31, -32, -33, -34, -35, -36, -37};
}

inline std::vector<std::vector<int>> GenerateSquareSmallMatrix() {
  return {{1, 2}, {3, 4}};
}
inline std::vector<int> GenerateSquareSmallExpected() {
  return {3, 4};
}

inline std::vector<std::vector<int>> GenerateVerticalMatrix() {
  return {{1, 2, 3, 4},     {5, 6, 7, 8},     {9, 10, 11, 12},  {13, 14, 15, 16}, {17, 18, 19, 20},
          {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}, {37, 38, 39, 40}};
}
inline std::vector<int> GenerateVerticalExpected() {
  return {37, 38, 39, 40};
}

inline std::vector<std::vector<int>> GenerateHorizontalMatrix() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
          {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
          {21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
          {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
          {41, 42, 43, 44, 45, 46, 47, 48, 49, 50}};
}
inline std::vector<int> GenerateHorizontalExpected() {
  return {41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
}

inline std::vector<std::vector<int>> GenerateCheckerboardMatrix() {
  return {{1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1},
          {1, -1, 1, -1, 1, -1, 1}, {-1, 1, -1, 1, -1, 1, -1}, {1, -1, 1, -1, 1, -1, 1}};
}
inline std::vector<int> GenerateCheckerboardExpected() {
  return {1, 1, 1, 1, 1, 1, 1};
}

inline std::vector<std::vector<int>> GenerateEmptyMatrix() {
  return {};
}
inline std::vector<std::vector<int>> GenerateZeroColumnsMatrix() {
  return std::vector<std::vector<int>>(5, std::vector<int>());
}
inline std::vector<std::vector<int>> GenerateNonRectangularMatrix() {
  return {{1, 2, 3}, {4, 5}, {6, 7, 8}};
}
inline std::vector<int> GenerateEmptyExpected() {
  return {};
}

inline std::vector<std::vector<int>> GenerateMatrixFunc(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantMatrix();
    case MatrixType::kSingleRow:
      return GenerateSingleRowMatrix();
    case MatrixType::kSingleCol:
      return GenerateSingleColMatrix();
    case MatrixType::kAllZeros:
      return GenerateAllZerosMatrix();
    case MatrixType::kConstant:
      return GenerateConstantMatrix();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstMatrix();
    case MatrixType::kMaxLast:
      return GenerateMaxLastMatrix();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleMatrix();
    case MatrixType::kAscending:
      return GenerateAscendingMatrix();
    case MatrixType::kDescending:
      return GenerateDescendingMatrix();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantMatrix();
    case MatrixType::kSparse:
      return GenerateSparseMatrix();
    case MatrixType::kNegative:
      return GenerateNegativeMatrix();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallMatrix();
    case MatrixType::kVertical:
      return GenerateVerticalMatrix();
    case MatrixType::kHorizontal:
      return GenerateHorizontalMatrix();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardMatrix();
    case MatrixType::kEmpty:
      return GenerateEmptyMatrix();
    case MatrixType::kZeroColumns:
      return GenerateZeroColumnsMatrix();
    case MatrixType::kNonRectangular:
      return GenerateNonRectangularMatrix();
  }
  return GenerateSingleConstantMatrix();
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:
      return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:
      return GenerateSingleColExpected();
    case MatrixType::kAllZeros:
      return GenerateAllZerosExpected();
    case MatrixType::kConstant:
      return GenerateConstantExpected();
    case MatrixType::kMaxFirst:
      return GenerateMaxFirstExpected();
    case MatrixType::kMaxLast:
      return GenerateMaxLastExpected();
    case MatrixType::kMaxMiddle:
      return GenerateMaxMiddleExpected();
    case MatrixType::kAscending:
      return GenerateAscendingExpected();
    case MatrixType::kDescending:
      return GenerateDescendingExpected();
    case MatrixType::kDiagonalDominant:
      return GenerateDiagonalDominantExpected();
    case MatrixType::kSparse:
      return GenerateSparseExpected();
    case MatrixType::kNegative:
      return GenerateNegativeExpected();
    case MatrixType::kSquareSmall:
      return GenerateSquareSmallExpected();
    case MatrixType::kVertical:
      return GenerateVerticalExpected();
    case MatrixType::kHorizontal:
      return GenerateHorizontalExpected();
    case MatrixType::kCheckerboard:
      return GenerateCheckerboardExpected();
    case MatrixType::kEmpty:
    case MatrixType::kZeroColumns:
    case MatrixType::kNonRectangular:
      return GenerateEmptyExpected();
  }
  return GenerateSingleConstantExpected();
}

}  // namespace tsibareva_e_matrix_column_max
```