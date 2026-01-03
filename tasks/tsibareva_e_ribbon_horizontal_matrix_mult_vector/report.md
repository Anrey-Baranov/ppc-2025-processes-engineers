# Ленточная горизонтальная схема - умножение матрицы на вектор

- Студент: Цибарева Екатерина Алексеевна, группа 3823Б1ПР1
- Технология: SEQ | MPI
- Вариант: 11

## 1. Введение
Задача умножения матрицы на вектор является одной из ключевых вычислительных процедур в линейной алгебре. Она составляет вычислительное ядро множества алгоритмов, включая итерационные методы решения систем линейных уравнений, методы вычисления собственных значений, а также алгоритмы в области машинного обучения и компьютерной графики. При работе с матрицами большого размера последовательные алгоритмы сталкиваются с ограничениями производительности, что делает актуальным применение параллельных вычислений.

Целью данной работы является реализация и сравнительный анализ последовательного и параллельного (на основе технологии MPI с ленточным горизонтальным разделением данных) алгоритмов умножения плотной матрицы на вектор.

## 2. Постановка задачи
**Описание задачи**
Вычислить результирующий вектор путем умножения заданной матрицы на заданный вектор.

**Входной тип данных** 
- Вектор значений типа int (представляющий матрицу, хранимую по строкам (row-major)).
- Количество строк.
- Количество столбцов.
- Заданный вектор значений типа int.

Удобство хранения в row-major порядке обусловлено в том числе необходимостью рассылки плоского массива в виде непрерывных интервалов памяти. Задача перевода из column-major в row-major порядок хранения матрицы может быть решена отдельно, вне последовательной и параллельной реализации решения задачи.

``` cpp
using InType = std::tuple<std::vector<int>, int, int, std::vector<int>>;
```

**Выходной тип данных** 
- Вектор значений типа int, представляющий результат умножения заданной матрицы на заданный вектор.

``` cpp
using OutType = std::vector<int>;
```

**Ограничения**
- матрица может состоять из повторяющихся значений либо содержать повторяющиеся значения, а также состоять как из положительных, так и отрицательных либо нулевых значений;
- результирующие вектора последовательной и параллельной реализаций алгоритма не должны различаться;
- для реализации параллельного алгоритма должен быть использован MPI;
- матрица может быть квадратной или прямоугольной;
- длина исходного вектора должна совпадать с количеством столбцов в матрице;
- исходный вектор и исходная матрица должны содержать данные одного типа;
- размер исходной матрицы должен соответствовать размерам, поданным во входном типе InType в качестве количества строк и столбцов;
- матрица может быть подана на вход алгоритма в не корректном формате, а именно быть пустой.

## 3. Описание базового алгоритма
**Основная идея**

Базовый, или последовательный, алгоритм выполняет умножение матрицы на вектор путем скалярного произведения каждой строки матрицы на вектор. Для получения i-го элемента результирующего вектора необходимо вычислить сумму произведений элементов i-й строки матрицы на соответствующие элементы вектора.

Конвейер выполнения как последовательного, так и параллельного алгоритма предполагает обязательную проверку на не пустоту матрицы и не пустоту её столбцов. 
``` cpp
if (rows_ == 0 || cols_ == 0) {
    return true;
}
```

**Шаги базового (последовательного) алгоритма обработки**: 

1. Инициализация матрицы, исходного вектора и исходных размеров матрицы.
2. Проверка на пустоту матрицы, причем штатной ситуацией для возращаемого значения пустых матриц считается возвращение пустого вектора.
3. Инициализация выходного вектора.
4. Алгоритм предполагает последовательный обход строк матрицы. Для каждой строки вычисляется скалярное произведение ее элементов на элементы входного вектора. Вычислительная сложность алгоритма составляет O(m * n), где m — количество строк, n — количество столбцов матрицы.

Частичный код базового (последовательного) алгоритма можно видеть ниже.

``` cpp
bool TsibarevaEMatrixColumnMaxSEQ::RunImpl() {
  ...
  auto &result_vector = GetOutput();

  for (int row = 0; row < rows_; ++row) {
    int sum = 0;
    for (int col = 0; col < cols_; ++col) {
      int matrix_idx = (row * cols_) + col;
      sum += input_matrix_[static_cast<size_t>(matrix_idx)] * input_vector_[static_cast<size_t>(col)];
    }
    result_vector[row] = sum;
  }

  return true;
}
```

## 4. Схема распараллеливания
**Модель распределения**

Для распараллеливания задачи используется схема ленточного горизонтального разделения данных, при котором матрица делится на полосы по строкам. Каждому процессу-исполнителю передается для обработки своя, непрерывная по строкам, часть матрицы (или лента).

Предполагается равномерное распределение строк, где первые remainder процессов получают на одну строку больше остальных. Распределение происходит в порядке возрастания ранга процесса: каждый процесс получает строки, последовательно идущие в исходной матрице, начиная с определенного смещения.

Исходный вектор является общим для всех вычислений и должен быть передан каждому процессу в полном объеме.

**Роли процессов**

Процесс 0 выступает в роли координатора:
- распределяет и рассылает данные матрицы в количестве определенного числа строк на процесс;
- собирает частичные результаты с процессов рангом 1 и выше;
- формирует результирующий вектор в соответствии с моделью рапределения строк между процессами;
- формирует и рассылает финальный результат.

Процессы рангом 1 и выше:
- выполняют локальные вычисления скалярных произведений назначенных и направленных им строк на общий (исходный) вектор;
- передают результаты локальных вычислений процессу с рангом 0.

**Коммуникационная схема**

**Шаги параллельного алгоритма обработки**: 

0. Рассылка исходного вектора (поскольку изначально является плоским std вектором и требует рассылки целиком на все процессы, может быть выполнена незамедлительно)
``` cpp
void TsibarevaERibbonHorizontalMatrixMultVectorMPI::BroadcastVector() {
  local_vector_.resize(cols_);
  MPI_Bcast(local_vector_.data(), cols_, MPI_INT, 0, MPI_COMM_WORLD);
}
```
1. Подготовительные действия на процессе 0: расчет смещений в исходном плоском массиве, представляющем матрицу, и рассылка массивов, содержащих смещения и количества элементов на процесс, - всем запущенным процессам. 

2. Рассылка данных на каждый процесс посредством MPI_Scatterv:
``` cpp
void TsibarevaERibbonHorizontalMatrixMultVectorMPI::ScatterMatrixData(int world_rank,
                                                                      const std::vector<int> &send_counts,
                                                                      const std::vector<int> &displacements) {
  local_flat_data_.resize(static_cast<size_t>(local_rows_) * cols_);
  MPI_Scatterv(world_rank == 0 ? input_matrix_.data() : nullptr,
               send_counts.data(),
               displacements.data(),
               MPI_INT,
               local_flat_data_.data(),
               static_cast<int>(local_flat_data_.size()),
               MPI_INT,
               0,
               MPI_COMM_WORLD);
}
```
3. Подсчет локальных значений вектора скалярного произведения по выделенным процессу строкам (лентам) на исходный вектор:
``` cpp
std::vector<int> TsibarevaERibbonHorizontalMatrixMultVectorMPI::CalculateMultiplyLocalPart() {
  std::vector<int> local_result(local_rows_, 0);

  for (int local_row = 0; local_row < local_rows_; local_row++) {
    int sum = 0;
    for (int col = 0; col < cols_; col++) {
      int matrix_idx = (local_row * cols_) + col;
      sum += local_flat_data_[matrix_idx] * local_vector_[col];
    }
    local_result[local_row] = sum;
  }

  return local_result;
}
```
4. Сбор результата на процессе 0 и синхронизация посредством MPI_Allgatherv (использование MPI_Allgatherv обеспечивает синхронизацию и наличие полного результата на всех процессах):
``` cpp
MPI_Allgatherv(local_result.data(),
               local_rows_,
               MPI_INT,
               global_result.data(),
               recv_counts.data(),
               displs.data(),
               MPI_INT,
               MPI_COMM_WORLD);
```

5. Установление GetOutput корректного значения результата.

**Особенности модели распределения и коммуникационной схемы**:
- распределение данных матрицы между процессами последовательно по строкам, топология "звезда" (нулевой процесс, как координатор, рассылает данные на все процессы);
- копирование исходного вектора в полном размере на каждый процесс;
- выполнение распределения, рассылок и формирования результата первично - на нулевом (координаторском) процессе;
- эффективный сбор и синхронизация данных на всех процессах одновременно.

Частичный код реализации алгоритма параллельной обработки можно видеть ниже.

``` cpp
bool TsibarevaERibbonHorizontalMatrixMultVectorMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  BroadcastMatrixDimensions();

  if (rows_ == 0 || cols_ == 0) {
    GetOutput() = std::vector<int>();
    return true;
  }

  BroadcastVector();

  int rows_base = rows_ / world_size;
  int remainder = rows_ % world_size;
  local_rows_ = rows_base + (world_rank < remainder ? 1 : 0);

  std::vector<int> send_counts;
  std::vector<int> displacements;
  PrepareScatterParameters(world_rank, world_size, send_counts, displacements);

  ScatterMatrixData(world_rank, send_counts, displacements);

  std::vector<int> local_result = CalculateMultiplyLocalPart();

  std::vector<int> recv_counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<int> global_result(rows_);

  int mdisplace = 0;
  for (int i = 0; i < world_size; i++) {
    int proc_rows = rows_base + (i < remainder ? 1 : 0);
    recv_counts[i] = proc_rows;
    displs[i] = mdisplace;
    mdisplace += proc_rows;
  }

  MPI_Allgatherv(local_result.data(), local_rows_, MPI_INT, global_result.data(), recv_counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}
```

## 5. Детали реализации

**Структура проекта**
``` text
tsibareva_e_ribbon_horizontal_matrix_mult_vector
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

- ops_seq.hpp - объявление класса TsibarevaERibbonHorizontalMatrixMultVectorSEQ, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl());

- ops_seq.cpp - реализация перегруженных методов:
    - RunImpl()            - базовый (последовательный) алгоритм умножения матрицы на вектор;
    - PreProcessingImpl()  - инициализирован выходной вектор, проверены случаи не корректного задания матрицы (пустота);
    - ValidationImpl()     - не предположено отдельной логики;
    - PostProcessingImpl() - не предположено отдельной логики.

2. MPI реализация (mpi):

- ops_mpi.hpp - объявление класса TsibarevaERibbonHorizontalMatrixMultVectorMPI, содержащего 4 метода, обязательных к переопределению (RunImpl(), PreProcessingImpl(), PostProcessingImpl(), ValidationImpl()), объявлен вектор input_matrix_ (исходная матрица), input_vector_ (исходный копируемый на все процессы вектор), local_flat_data_ (локальные данные (ленты) каждого процесса);

- ops_mpi.cpp - реализация методов:
    - ValidationImpl()              - не предположено отдельной логики;
    - PreProcessingImpl()           - не предположено отдельной логики;
    - RunImpl()                     - инициализирован выходной вектор, проверены случаи не корректного задания матрицы (пустота), реализован параллельный алгоритм умножения матрицы на вектор, распределяющий содержательные действия между процессами;
    - PostProcessingImpl()          - не предположено отдельной логики;
    - PrepareScatterParameters(...) - подготавливает и рассылает вектора смещений в исходном плоском векторе для последующей рассылки данных с нулевого процесса по остальным запущенным;
    - ScatterMatrixData(...)        - рассылает данные исходной матрицы в количестве нескольких строк на запущенный процесс;
    - BroadcastMatrixDimensions()   - рассылает размеры матрицы (rows_, cols_) со процесса 0 на все процессы;
    - BroadcastVector()             - рассылает исходный вектор целиком со процесса 0 на все процессы;
    - CalculateMultiplyLocalPart()  - вычисляет локальную часть результирующего вектора путем скалярного умножения полученных строк матрицы на исходный вектор.

**Тестирование и формирование входных данных**

3. Общие компоненты (common):
- common.hpp содержит:
    - объявление входного (InType), выходного (OutType) типа, типа базовой задачи (BaseTask) и тестовых классов (TestTask); необходимо отметить, что входные данные представлены в формате плоского вектора, выходные - плоского результирующего вектора; 
    ``` cpp
    using InType = std::tuple<std::vector<int>, int, int, std::vector<int>>;
    using OutType = std::vector<int>;
    using TestType = std::tuple<MatrixType, std::string>;
    using BaseTask = ppc::task::Task<InType, OutType>;
    ``` 
    - перечисление типов матриц, доступных для генерации (enum MatrixType), более подробно описанных в пункте 7.1 "Корректность"; необходимо отметить, что для каждого типа матрицы был предположен свой размер и свой тип заданного исходного вектора;
    - функции формирования матриц 11 типов в соответствии с перечислением;
    - функцию задания матрицы GenerateMatrixFunc, в качестве аргументов принимающую тип генерируемой матрицы, и принимающей решение о вызове соответствующей функции генерации матрицы; GenerateMatrixFunc возвращает std::tuple, состоящий из подготовленных значений исходной матрицы, её размеров и исходного вектора;
    - функцию GenerateExpectedOutput, задающую эталонный результирующий вектор для каждого типа задачи.

Например, подготовка данных матрицы, размеров и исходного вектора может выглядеть так, как представлено ниже.
``` cpp
inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMixedSigns() {
  return {{1, -2, 3, -4, -5, 6, -7, 8}, 2, 4, {1, -1, 1, -1}};
}
inline std::vector<int> GenerateMixedSignsExpected() {
  return {10, -26};
}
```

4. Функциональные тестовые файлы (tests/funtional):
- main.cpp - содержит:
    - объявление класса TsibarevaERunFuncTestsProcesses;
    - переопределение функции PrintTestParam таким образом, что печать тестов содержит краткое описание её типа, включающее размерность;
    - переопределение функции SetUp таким образом, что исходная матрица может быть получена как GenerateMatrixFunc(<тип матрицы 1 из 11 перечисленных>), а ожидаемые данные - как GenerateExpectedOutput(<тот же тип матрицы>);
    ``` cpp
    MatrixType matrix_type = std::get<0>(params);
    input_data_            = GenerateMatrixFunc(matrix_type);
    expected_output_       = GenerateExpectedOutput(matrix_type);
    ```
    при этом каждому типу матрицы заведомо соответствуют её размеры и исходный вектор;
    - инициализацию и заполнение массива аргументов 12 тестовыми случаями, выполнимыми как для последовательной, так и для параллельной реализации алгоритма; тестовые аргументы покрывают такие случаи, как умножение единичной матрицы, матрицы, состоящая из одной строки/столбца, умножение квадратной матрицы, прямоугольной матрицы, у которой количество столбцов/строк превышает количество строк/столбцов соответственно; тестовые аргументы покрывают ситуации, в которых на вход алгоритму передана пустая матрица; во всех случаях производится тестирование на векторе, соответствующем исходной матрице.

5. Тестовые файлы производительности (tests/performance):
- main.cpp - содержит:
    - объявление класса TsibarevaERunPerfTestProcesses;
    - переопределение функции SetUp таким образом, что исходная матрица и исходный вектор в ней формируются на фиксированное количество элементов, размером 20000 на 20000 для матрицы и 20000 для вектора элементов соответственно для запуска Github Actions и 6000*6000 и 6000 соответственно - для локального запуска. 

Для тестирования производительности были сгенерированы псевдослучайные наборы данных как для исходной матрицы, так и для исходного вектора. Использование рандомной генерации данных предположено не было.

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
Для проверки корректности выполнения алгоритмами вычисления произведения матрицы на вектор были использованы предварительно подготовленные матрицы и эталонные вектора 11 типов. Каждый тип был прописан в enum MatrixType и предполагает как покрытие различных размерностей матриц, так и различных типов их формирования. Так, тестовые данные (матрицы и соответствующие исходные вектора) могут быть следующих типов:

| Тип матрицы	    |  Краткое описание                                 |
| ------------------|---------------------------------------------------|
| kSingleConstant	| Матрица 1×1 (один элемент)                        |
| kSingleRow	    | Матрица 1×N (одна строка)                         |
| kSingleCol	    | Матрица M×1 (один столбец)                        |
| kEmpty	        | Пустая матрица 0×0                                |
| kSquare	        | Квадратная матрица N×N                            |
| kMoreRows	        | Прямоугольная матрица (строк больше чем столбцов) |
| kMoreCols	        | Прямоугольная матрица (столбцов больше чем строк) |
| kAllZeros	        | Матрица, состоящая только из нулей                |
| kPositive	        | Матрица с положительными элементами               |
| kNegative	        | Матрица с отрицательными элементами               |
| kMixedSigns	    | Матрица со смешанными знаками элементов           |

Был предположен отказ от генерации матриц посредством функций в пользу формирования изначально предопределенных наборов данных для матриц и эталонных векторов. Эталонные вектора, возвращаемые по аргументу типа соответствующего типу формирования матрицы, содержат результаты умножения этих матриц на соответствующие исходные вектора. 

Полный код формирования тестовых случаев функционального тестирования можно видеть в приложении 1.

### 7.2 Производительность
Тесты производительности были проведены локально на 36 миллионах (для исходной матрицы) и 6 тысячах (для исходного вектора) элементов.

### Режим Pipeline
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.093    | 1.00      | N/A           |
| 2 (MPI)   | 0.036    | 2.58      | 129%          |
| 3 (MPI)   | 0.032    | 2.91      |  97%          |
| 4 (MPI)   | 0.036    | 2.58      |  65%          |

### Режим Task Run  
| Процессов | Время, с | Ускорение | Эффективность |
|-----------|----------|-----------|---------------|
| 1 (SEQ)   | 0.110    | 1.00      | N/A           |
| 2 (MPI)   | 0.033    | 3.33      | 167%          |
| 3 (MPI)   | 0.043    | 2.56      |  85%          |
| 4 (MPI)   | 0.039    | 2.82      |  71%          |

MPI реализация показала ускорение до 3 раз относительно последовательной реализации алгоритма умножения матрицы на вектор. Наблюдается линейное ухудшение эффективности от 129% до 65%, обусловленное ростом накладных расходов с увеличением количества процессов.

## 8. Выводы
Задача реализации алгоритма умножения матрицы на вектор была решена последовательным образом и параллельным образом при использовании MPI. Были предложены различные способы формирования исходных данных для функциональных тестов и тестов производительности. А также выполнены замеры времени выполнения последовательного и параллельного алгоритмов с последующим высчитыванием эффективности распараллеливания вычислений.

Параллельная реализация демонстрирует ожидаемое ускорение на больших размерах матриц.

## 9. Источники

1. **Технологии параллельного программирования MPI и OpenMP** // А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.
2. **Инструменты параллельного программирования в системах с общей памятью: Учебное пособие.** // Корняков К.В., Мееров И.Б., Сиднев А.А., Сысоев А.В., Шишков А.В., - Нижний Новгород: Изд-во Нижегородского госуниверситета, 2010. - 202 с.
3. **Справочник по MPI** // URL: https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-reference, 2023 (дата обращения: 15.11.2025).
4. **Open MPI: Open Source High Performance Computing** // URL: https://www-lb.open-mpi.org/doc/v4.1, 2025 (дата обращения: 15.11.2025).

### 10. Приложения

## Приложение 1: генерация данных для функциональных тестов
``` cpp
enum class MatrixType : std::uint8_t {
  kSingleConstant,  // 1x1
  kSingleRow,       // 1xN
  kSingleCol,       // Mx1
  kEmpty,           // 0x0
  kSquare,          // NxN
  kMoreRows,        // M>N
  kMoreCols,        // M<N
  kAllZeros,        // Все нули
  kPositive,        // Положительные
  kNegative,        // Отрицательные
  kMixedSigns       // Смешанные знаки
};

using InType = std::tuple<std::vector<int>, int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<MatrixType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleConstant() {
  return {{5}, 1, 1, {2}};
}
inline std::vector<int> GenerateSingleConstantExpected() {
  return {10};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleRow() {
  return {{1, 2, 3, 4, 5}, 1, 5, {1, 2, 3, 2, 1}};
}
inline std::vector<int> GenerateSingleRowExpected() {
  return {27};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSingleCol() {
  return {{1, 2, 3, 4}, 4, 1, {3}};
}
inline std::vector<int> GenerateSingleColExpected() {
  return {3, 6, 9, 12};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateEmpty() {
  return {{}, 0, 0, {}};
}
inline std::vector<int> GenerateEmptyExpected() {
  return {};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateSquare() {
  return {{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3, {1, 1, 1}};
}
inline std::vector<int> GenerateSquareExpected() {
  return {1, 2, 3};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMoreRows() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2, {2, 1}};
}
inline std::vector<int> GenerateMoreRowsExpected() {
  return {4, 10, 16, 22, 28};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMoreCols() {
  return {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 2, 5, {1, 0, 1, 0, 1}};
}
inline std::vector<int> GenerateMoreColsExpected() {
  return {9, 24};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateAllZeros() {
  std::vector<int> matrix(16, 0);
  return {matrix, 4, 4, {1, 2, 3, 4}};
}
inline std::vector<int> GenerateAllZerosExpected() {
  return {0, 0, 0, 0};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GeneratePositive() {
  return {{2, 2, 2, 3, 3, 3, 4, 4, 4}, 3, 3, {1, 1, 1}};
}
inline std::vector<int> GeneratePositiveExpected() {
  return {6, 9, 12};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateNegative() {
  return {{-1, -2, -3, -4, -5, -6, -7, -8, -9}, 3, 3, {-1, -1, -1}};
}
inline std::vector<int> GenerateNegativeExpected() {
  return {6, 15, 24};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateMixedSigns() {
  return {{1, -2, 3, -4, -5, 6, -7, 8}, 2, 4, {1, -1, 1, -1}};
}
inline std::vector<int> GenerateMixedSignsExpected() {
  return {10, -26};
}

inline std::tuple<std::vector<int>, int, int, std::vector<int>> GenerateTestData(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstant();
    case MatrixType::kSingleRow:
      return GenerateSingleRow();
    case MatrixType::kSingleCol:
      return GenerateSingleCol();
    case MatrixType::kEmpty:
      return GenerateEmpty();
    case MatrixType::kSquare:
      return GenerateSquare();
    case MatrixType::kMoreRows:
      return GenerateMoreRows();
    case MatrixType::kMoreCols:
      return GenerateMoreCols();
    case MatrixType::kAllZeros:
      return GenerateAllZeros();
    case MatrixType::kPositive:
      return GeneratePositive();
    case MatrixType::kNegative:
      return GenerateNegative();
    case MatrixType::kMixedSigns:
      return GenerateMixedSigns();
    default:
      return GenerateSingleConstant();
  }
}

inline std::vector<int> GenerateExpectedOutput(MatrixType type) {
  switch (type) {
    case MatrixType::kSingleConstant:
      return GenerateSingleConstantExpected();
    case MatrixType::kSingleRow:
      return GenerateSingleRowExpected();
    case MatrixType::kSingleCol:
      return GenerateSingleColExpected();
    case MatrixType::kEmpty:
      return GenerateEmptyExpected();
    case MatrixType::kSquare:
      return GenerateSquareExpected();
    case MatrixType::kMoreRows:
      return GenerateMoreRowsExpected();
    case MatrixType::kMoreCols:
      return GenerateMoreColsExpected();
    case MatrixType::kAllZeros:
      return GenerateAllZerosExpected();
    case MatrixType::kPositive:
      return GeneratePositiveExpected();
    case MatrixType::kNegative:
      return GenerateNegativeExpected();
    case MatrixType::kMixedSigns:
      return GenerateMixedSignsExpected();
    default:
      return GenerateSingleConstantExpected();
  }
}
```
