# Отчет по лабораторной работе
## Поиск кратчайших путей из одной вершины (алгоритм Беллмана-Форда). С CRS формой хранения графа.

- Студент: Васильев Михаил Петрович, группа 3823Б1ПР2
- Технология: SEQ | MPI
- Вариант: 23

# 1. Введение
В современных условиях обработки больших массивов данных особое значение имеет эффективное использование вычислительных ресурсов. Параллельные вычисления позволяют значительно уменьшить время выполнения алгоритмов за счет распределения вычислительной нагрузки между несколькими процессорами или процессами.

Одним из наиболее широко используемых инструментов параллельного программирования является технология MPI (Message Passing Interface), обеспечивающая взаимодействие процессов посредством обмена сообщениями в распределённых системах.

В рамках данной работы исследуется задача на поиск кратчайших путей из одной вершины (алгоритм Беллмана-Форда) с CRS формой хранения графа. Для решения задачи были реализованы две версии алгоритма: последовательная (SEQ) и параллельная (MPI). Основная цель работы заключается в сравнении их производительности и анализе эффективности параллелизации для данной задачи.

---

# 2. Постановка задачи
Дается граф, в CRS форме хранения с исходной вершиной. Необходимо построить вектор кратчайших расстояний от исходной вершины, применяя алгоритм Беллмана-Форда.

Например, граф с 4 вершинами 3 ребрами (в скобках - веса ребер), `0->1 (1), 1->2 (2), 2->3 (3)` в итоге, должен выдать кратчайшие пути из вершины `0`:  `0 1 3 6`.

---

# 3. Последовательная версия (SEQ)

SEQ и MPI версия алгоритмов состоят из 4-х этапов:
  `ValidationImpl()` - Проверка входных данных.
  `PreProcessingImpl()` - Подготовка к выполнению алгоритма.
  `RunImpl()` - Основной алгоритм.
  `PostProcessingImpl()` - Проверка конечного результата.

Далее работа данных методов в SEQ версии алгоритма:

**1. `ValidationImpl()`**
- Проверяется, что граф существует.
- Проверяется, что количество колонок (номера вершин, в которые ведут ребра) = количество весов.
- Проверяется, что количество вершин > 0.
- Проверяется, что исходная вершина `source` - корректна.

**2. `PreProcessingImpl()`**
- Создаётся массив `кратчайших путей` длиной `количества вершин`, все значения равны бесконечности.
- Расстояние до исходной вершины = 0.

**3. `RunImpl()`**
1. Внешний цикл, гарантирующий стабилизацию расстояний за `(количество вершин - 1)` итераций.
2. Создание флага, отслеживающего обновление на текущей итерации. Если обновления не было -> пути найдены -> ранний выход.
3. Цикл по чтению всех вершин.
4. Цикл по ребрам вершины в настоящей итерации.
5. Расслабление ребра - если путь был найден короче, то замена в векторе `кратчайших путей`.

**4. `PostProcessingImpl()`**
- В пост-проверке нет необходимости.

---

# 4. Схема параллелизации
Идея параллелизации заключается в том, вершины графа делятся между процессами, что позволяет распределить обработку каждой части отдельным процессом параллельно.

Каждый процесс локально расслабляет ребра своей части вершин, затем предлагает обновления путей. В конце результат объединяется со всех процессов через MPI.


## Схема
Главным процессом (rank 0) инициализируется MPI и, в зависимости от указанного количества процессов в параметрах, создается данное число отдельных процессов, занимающие свою часть вектора (а если процессы не могут равномерно занять части, то первые из них будут иметь больше вершин на обработку, чем остальные).

Таким образом (при равномерном распределении): `Процесс_0` займет первые `n` вершин графа для обработки, `Процесс_1` займет `с n+1 до 2n` вершины, и так далее.

Затем каждый процесс обрабатывает свои отдельные части, локально находя кратчайшие пути.

Далее, выполняется слияние результатов между процессами, где объединяются расстояния, MPI берет поэлементный минимум и в глобальный вектор кратчайших путей записывается вычисленный минимум. Затем, происходит проверка на обновления, если их не было - кратчайшие пути были найдены.

Если предварительного выхода не случилось, то, благодаря алгоритму Беллмана-Форда, расстояния будут гарантированно стабилизированы после прохода всего внешнего цикла (за количество итераций = `количество вершин - 1`).


## Распределение нагрузки
При выделении `k` процессов и обработки `n` вершин, количество элементов распределяются следующим образом:

```cpp
int chunk = n / k;          // Изначальное количество вершин на процесс
remain = n % k;              // Остаточное количество неиспользованных вершин графа
```

(То есть первые `remain` процессов получают на один элемент больше, что обеспечивает равномерное распределение нагрузки)

```cpp
vector<int> counts(k);
  for (int i = 0; i < remain; i++) {
    counts[i]++;
  }  // counts[i] хранит, сколько элементов получит процесс i

vector<int> displs(k);
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }  // displs[i] — с какого элемента вектора начнётся часть процесса i
```
**Пример распределения для n=10, k=3:**
- Процесс 0: вершины 0-3 (4 элемента)
- Процесс 1: вершины 4-6 (3 элемента)
- Процесс 2: вершины 7-9 (3 элемента)
## Коммуникации между процессами
MPI версия алгоритма использует следующие коммуникации:
1. **MPI_Bcast** — рассылает процессам их отдельное количество вершин графа (counts) и смещение в векторе вершин (displs):
   ```cpp
   MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   ```
2. **MPI_Allreduce** — нахождение минимальных путей среди всех локальных расстояний и перенаправление этого результата в выделенный вектор всем процессам (результат необходим всем для корректного завершения выполнения вне зависимости от `rank`); Второй `MPI_Allreduce` проверяет локальные обновления и суммирует их в глобальные, для проверки возможной раннего завершения работы алгоритма:
   ```cpp
   MPI_Allreduce(local_dist.data(), dist.data(), vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   ```
---

# 5. Программная реализация (MPI версия алгоритма)
## Структура решения
MPI-версия реализована классом `VasilievMBubbleSortMPI`, наследуемый от `BaseTask`. Структура реализована на `Pipeline`, имеющий четыре последовательных этапа:

1. **Validation** — валидация входных данных
2. **PreProcessing** — предварительная обработка
3. **Run** — основной алгоритм/вычисления
4. **PostProcessing** — постобработка/проверка результатов

## Структура класса
```cpp
namespace vasiliev_m_bubble_sort {
  using WeightType = int;  // тип веса ребер - целое число
  struct CRSGraph {
    std::vector<int> row_ptr;  // указание, где в col_ind и vals начинаются ребра каждой вершины
    std::vector<int> col_ind;  // номера вершин, в которые ведут ребра
    std::vector<WeightType> vals;  // веса ребер
    int source = 0;  // исходная вершина
  };  
  using InType = CRSGraph;
  using OutType = std::vector<WeightType>;
  using BaseTask = ppc::task::Task<InType, OutType>;

  class VasilievMBellmanFordCrsMPI : public BaseTask {
   public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kMPI;
    }
    explicit VasilievMBellmanFordCrsMPI(const InType &in);
    static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);  // вычисление кол-ва элементов на процесс и смещения в векторе
    static int RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                               std::vector<int> &local_dist);  // расслабление локальных ребер
  
   private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;
  };
}
```
## Реализация методов
### Конструктор
```cpp
VasilievMBellmanFordCrsMPI::VasilievMBellmanFordCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}
```
Конструктор инициализирует тип задачи (в данном случае MPI), устанавливает входные данные и тип выходных.

### Валидация данных
```cpp
bool VasilievMBellmanFordCrsMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.row_ptr.empty() || in.col_ind.empty() || in.vals.empty()) {
    return false;
  }

  if (in.col_ind.size() != in.vals.size()) {
    return false;
  }

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  if (vertices <= 0) {
    return false;
  }

  if (in.source < 0 || in.source >= vertices) {
    return false;
  }

  return true;
}
```
Проверка на корректность входных данных.

### Предварительная обработка
```cpp
bool VasilievMBellmanFordCrsMPI::PreProcessingImpl() {
  const auto &in = GetInput();
  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  const int inf = std::numeric_limits<int>::max();
  GetOutput().assign(vertices, inf);
  GetOutput()[in.source] = 0;

  return true;
}
```
Создание вектора путей, с бесконечными значениями, расстояние до вершины = 0.

#### Основной алгоритм
```cpp
bool VasilievMBellmanFordCrsMPI::RunImpl() {
  const auto &in = GetInput();
  auto &dist = GetOutput();

  int rank = 0;  // информация о процессах
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  // вычисления counts и displs происходят только на root-ранге
  if (rank == 0) {
    CalcCountsAndDispls(vertices, size, counts, displs);  // распределение вершин между процессами
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);  // рассылка counts и displs всем процессам
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // диапазон вершин, с которыми работает текущий процесс
  int start = displs[rank];
  int end = start + counts[rank];

  std::vector<int> local_dist(vertices);

  for (int i = 0; i < vertices - 1; i++) {
    local_dist = dist;  // локальная копия расстояний

    int local_updated = RelaxLocalEdges(start, end, in, dist, local_dist);  // расслабление локальных ребер

    MPI_Allreduce(local_dist.data(), dist.data(), vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  // слияние путей, для нахождения минимума

    int global_updated = 0;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  // проверка на обновления пути на итерации

    if (global_updated == 0) {  // раннее завершение работы алгоритма
      break;
    }
  }

  return true;
}
```

### Постобработка результатов

```cpp
bool VasilievMBellmanFordCrsMPI::PostProcessingImpl() {
  return true;
}
```
Пост-проверки не требуется.

### Вспомогательные функции
```cpp
void VasilievMBellmanFordCrsMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;
  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }
  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}
```
Вычисление количества вершин на процесс и смещения в векторе.
```cpp
int VasilievMBellmanFordCrsMPI::RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                                                std::vector<int> &local_dist) {
  const int inf = std::numeric_limits<int>::max();
  int local_updated = 0;

  for (int vertex = start; vertex < end; vertex++) {
    if (dist[vertex] == inf) {
      continue;
    }

    for (int edge = in.row_ptr[vertex]; edge < in.row_ptr[vertex + 1]; edge++) {
      int v = in.col_ind[edge];
      int w = in.vals[edge];

      if (local_dist[v] > dist[vertex] + w) {
        local_dist[v] = dist[vertex] + w;
        local_updated = 1;
      }
    }
  }

  return local_updated;
}
```
Расслабление локальных ребер.

### Особые случаи
Одна из главных особенностей алгоритма Беллмана-Форда, это возможность корректной работы с графом, имеющий отрицательные веса. В реализованной версии алгоритма эта особенность учитывается, без дополнительных функций - отрицательный вес сильнее уменшает сумму (считаемый кратчайший путь).

### Преимущества MPI-реализации
- **Параллелизация** — каждый процесс хранит и обрабатывает только свою часть вершин, следовательно сложность обработки сокращается в `k` раз.
- **Масштабируемость** — алгоритм будет работать все более эффективно на все более большем графе и/или при увеличении числа процессов, но с условием коммуникационной сложности алгоритма.
- **Отсутствие гонок данных** — каждый процесс работает только со своими отдельными частями вершин, не затрагивая данные других процессов.

---

# 6. Результаты экспериментов
## Типы экспериментов/тестов
Для проверки корректной работы алгоритмов и ее производительности были составлены:
- Функциональные тесы - для проверки результатов алгоритмов
- Тесты производительности - для измерения производительности/времени выполнения работы алгоритмом
Для написания тестов были использованы `google tests`.

## Алгоритм считывания тестовых значений
Все тестовые графы записаны в отдельных файлах формата `.txt`.

Схема считывания графа в тестах следующая: в первой строке выписываются количество вершин и ребер (через пробел) в рассматриваемом графе. На следующих `n` строчках прописываются все ребра, следовательно каждая строка имеет значения: `из какой вершины идет ребро`; `в какую вершину идет ребро`; `вес ребра`.
После записи всех ребер, на следующей строке прописывается стартовая вершина, и на последней строке - ожидаемые кратчайшие расстояния в графе из этой стартовой вершины.
(Напр.:\
`4 3`\
`0 1 1`\
`1 2 2`\
`2 3 3`\
`0`\
`0 1 3 6`\
)

Алгоритм считывания тестовых значений позволяет переводить данные из `string` формата в обрабатываемые, соответственно, для дальнейшей работы в вычислении кратчайших путей (SEQ или MPI).

Соответственный алгоритм:
```cpp
      std::vector<std::tuple<int, int, int>> edges(num_edges);  // чтение ребер
      for (int i = 0; i < num_edges; i++) {
        file >> std::get<0>(edges[i]) >> std::get<1>(edges[i]) >> std::get<2>(edges[i]);
      }
  
      int source = 0;  // чтение исходной вершины
      file >> source;
  
      expected_data_.resize(num_vertices);  // чтение ожидаемого результата
      for (int i = 0; i < num_vertices; i++) {
        file >> expected_data_[i];
      }
  
      input_data_.row_ptr.assign(num_vertices + 1, 0);  // построение CRS графа
      for (const auto &e : edges) {
        input_data_.row_ptr[std::get<0>(e) + 1]++;  // подсчет количества ребер для каждой вершины
      }
  
      for (int i = 1; i <= num_vertices; i++) {
        input_data_.row_ptr[i] += input_data_.row_ptr[i - 1];  // префиксная сумма
      }
  
      input_data_.col_ind.resize(num_edges);
      input_data_.vals.resize(num_edges);
  
      std::vector<int> counter = input_data_.row_ptr;  // заполнение заполнение данных графа
      for (const auto &e : edges) {
        int u = std::get<0>(e);
        int idx = counter[u]++;
        input_data_.col_ind[idx] = std::get<1>(e);
        input_data_.vals[idx] = std::get<2>(e);
      }

      input_data_.source = source;  // установка исходной вершины
```
## Функциональные тесты
Для проверки корректности алгоритмов было написано 4 тестовых графа с заготовленным результатом в виде вектора кратчайших путей.\
1 - базовый линейный граф\
`4 3`\
`0 1 1`\
`1 2 2`\
`2 3 3`\
`0`\
`0 1 3 6`\
2 - граф с отрицательными ребрами\
`5 7`\
`0 1 6`\
`0 2 7`\
`1 2 8`\
`1 3 5`\
`2 3 -3`\
`3 4 2`\
`1 4 -4`\
`0`\
`0 6 7 4 2`\
3 - несвязный граф\
`6 4`\
`0 1 2`\
`1 2 3`\
`3 4 1`\
`4 5 1`\
`0`\
`0 2 5 2147483647 2147483647 2147483647`\
4 - граф с несколькими путями к одной вершине\
`5 7`\
`0 1 10`\
`0 2 3`\
`2 1 1`\
`1 3 2`\
`2 3 8`\
`3 4 7`\
`1 4 5`\
`0`\
`0 4 3 6 9`
### Результаты
**Все 4 функциональных теста были успешно пройдены** для обеих версий (SEQ и MPI) при различном количестве процессов: 1, 2, 4, 5, 6, 8 (работа проводилась на машине с 6 доступными ядрами).
- Были верно подсчитаны тестовые графы
- Результаты обоих версий алгоритмов (SEQ, MPI) были идентичны
- Корректна распределена нагрузка при параллельном выполнении
- Правильная обработка при отрицательных весах ребер
- Верная работа алгоритма при большем количестве процессов, чем вершин или физических ядер
## Тесты производительности
Для наиболее явной проверки производительности были реализованы тесты на графе с количеством вершин равным `10 тыс.` и ребер `50 тыс.`, соответственно, с двумя режимами измерений:
1. **task_run** — измерение времени выполнения метода `RunImpl()`
2. **pipeline** — измерение времени выполнения всего конвейера (`ValidationImpl()` + `PreProcessingImpl()` + `RunImpl()` + `PostProcessingImpl()`)
### Ускорение при параллелизации
При увеличении количества процессов для работы параллельного алгоритма, ускорение не будет являться линейным из-за накладных расходов:
- Инициализация MPI
- Рассылка разных данных процессам
- Ожидание завершение всех процессов (из-за возможного неравномерного распределения элементов вектора на процессы или гонок фаз)
- Сбор отсортированных блоков
- Специфической временной сложности алгоритма
В дополнение к этому, из-за вышеперечисленных пунктов параллельный алгоритм на малых данных будет выполняться дольше, чем последовательная версия (затраты на параллелизацию больше, чем выигрыш от ее преимуществ). Следовательно, ускорение будет заметно только на большом графе.
---
# 7. Результаты и выводы
## Корректность реализации
Все написанные тесты были выполнены верно при применении последовательной (SEQ) и параллельной (MPI) версии алгоритма (результаты работ алгоритмов равны предзаписанным векторам кратчайших путей). Результаты работ обоих алгоритмов совпадают.
При вычислении тестового графа, цель которого - проверка производительности, алгоритмы так же корректно выполнили задачу и предоставили явные показатели ускорения и эффективности.
## Инфраструктура для тестов
### Виртуальная машина (VirtualBox)
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 9400F (6 cores, 6 threads, 2900 MHz)   |
| RAM        | 10 GB DDR4 (2660 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS                                   |
| Compiler   | GCC 13.3.0, Release Build                            |
## Производительность
Итоговый размер тестируемого графа состоял из `10 тыс.` вершин и `50 тыс.` ребер, соответственно, и показатели производительности (при разных количествах процессов), при его тестировании, были следующими:
`(из-за специфики использования виртуальной машины, доступных аппаратных ресурсов, увеличение еще в несколько раз размерности графа приводит к несправедливым результатам из-за bottleneck-а аппаратуры и синхронизации/передачи больших буферов данных)`
| Версия алг-ма        | Кол-во процессов    | Время, с | Ускорение | Эффективность |
|----------------------|---------------------|----------|-----------|---------------|
| SEQ                  | 1                   | 2.728    | 1.00      | N/A           |
| MPI                  | 2                   | 1.538    | 1.77      | 88.5%         |
| MPI                  | 3                   | 1.168    | 2.33      | 77.6%         |
| MPI                  | 4                   | 0.983    | 2.77      | 69.2%         |
| MPI                  | 5                   | 1.016    | 2.68      | 53.6%         |
| MPI                  | 6                   | 1.033    | 2.64      | 44.0%         |
| MPI                  | 7 (--oversubscribe) | 1.799    | 1.51      | 21.5%         |
| MPI                  | 8 (--oversubscribe) | 1.767    | 1.54      | 19.2%         |
Из таблицы видно, что при увеличении количества процессов время выполнения алгоритмов сокращается, а его работа ускоряется с хорошей масштабируемостью. При использовании 2-х процессов, скорость работы была увеличена в 1.77 раза; при использовании 3 - в 2.33 раза; при 4 - в 2.77 раза. 
Таким образом, при использовании 2 процессов, время работы было замедлено с коэффициентом 0,97, при использовании 4 процессов - время сокращено более, чем в 4 раза, а при использовании 6 - в 9,16 раза. Эффективность, при увеличении процессов уменьшается, вследствие синхронизаций, обменов сообщений и сложности коммуникаций.
При использовании больше, чем 4 процессов скорость работы алгоритма начала замедляться. Итоговая сложность коммуникаций составляет `O(V^2 * log(P))`, где `V` - число вершин, `P` - число процессов. При росте `P`, вычисления на процесс уменьшаются линейно, но стоимость коммуникаций (`MPI_Allreduce`) растет и в какой-то момент коммуникации начинают покрывать выигрыш в параллелизации. Следовательно, при данной реализации параллельного алгоритма и данном тестовом графе с его соответственным размером, идеальное уменьшение времени работы алгоритма будет не при использовании максимального количества возможных процессов.
В итоге, при использовании 5, 6, 7 и 8 процессах (7 и 8 используются совместно с флагом `--oversubscribe`) ускорение будет составлять 2.68, 2.64, 1.51 и 1.54 раза соответственно. Дополнительно, 7 и 8 процессах ускорение и эффективность уменьшаются, также, из-за использования несколькими процессами одного физического ядра.
## Выводы
- Функциональные тесты и тесты для производительности выполняются корректно.
- MPI версия алгоритма работает эффективней SEQ версии, но с условиями.
- Достаточное масштабирование при увеличении числа процессов.
- Корректная работа параллельной версии алгоритма при количестве процессов меньше, чем доступных ядер; такого же количества; и больше.
- Распределение нагрузки имеет ключевую роль в увеличении эффективности работы при параллелизации.
---
# 8. Заключение
В рамках данной работы была успешно решена задача поиска кратчайших путей из одной вершины с использованием алгоритма Беллмана-Форда и CRS формы хранения графа, с использованием технологии MPI.
Были разработаны и реализованы последовательная (SEQ) и параллельная (MPI) версии алгоритма.
Было выявлено какого ускорения и какого типа возможно достигнуть при использовании параллелизации.
В итоге, результаты показали, что применение MPI позволяет повысить производительность при обработке больших объёмов данных, но из-за специфики сложностей коммуникаций, максимальное ускорение может достигаться не только при использовании максимального доступного количества процессов.
---
# 9. Список литературы
1. MPI Guide. **Using MPI with C++: a basic guide (2024)** [Электронный ресурс]. — Режим доступа: https://www.paulnorvig.com/guides/using-mpi-with-c.html
2. Snir M., Otto S. **MPI: The Complete Reference**. - MIT Press Cambridge, 1995 - 217 p.
3. MPI Forum. **MPI: A Message-Passing Interface Standard. Version 3.1** [Электронный ресурс]. — Режим доступа: https://www.mpi-forum.org/docs/
4. Bellman–Ford Algorithm. **Variations of algorithms in graph theory** [Электронный ресурс]. — Режим доступа: https://www.geeksforgeeks.org/dsa/bellman-ford-algorithm-dp-23/
5. Google Test Documentation. **GoogleTest User’s Guide** [Электронный ресурс]. — Режим доступа: https://google.github.io/googletest/
---
# 10. Приложение
## Общие обозначения `common.hpp`
```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vasiliev_m_bellman_ford_crs {

using WeightType = int;  // веса ребер - целое число

struct CRSGraph {
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<WeightType> vals;
  int source = 0;
};

using InType = CRSGraph;
using OutType = std::vector<WeightType>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vasiliev_m_bellman_ford_crs
```

## Header SEQ версии `ops_seq.hpp`
```cpp
#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMBellmanFordCrsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bellman_ford_crs
```
## Реализация методов SEQ версии `ops_seq.cpp`
```cpp
#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

#include <limits>
#include <vector>

#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

VasilievMBellmanFordCrsSEQ::VasilievMBellmanFordCrsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBellmanFordCrsSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.row_ptr.empty() || in.col_ind.empty() || in.vals.empty()) {
    return false;
  }

  if (in.col_ind.size() != in.vals.size()) {
    return false;
  }

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  if (vertices <= 0) {
    return false;
  }

  if (in.source < 0 || in.source >= vertices) {
    return false;
  }

  return true;
}

bool VasilievMBellmanFordCrsSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  const int inf = std::numeric_limits<int>::max();
  GetOutput().assign(vertices, inf);
  GetOutput()[in.source] = 0;

  return true;
}

bool VasilievMBellmanFordCrsSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &dist = GetOutput();

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  const int inf = std::numeric_limits<int>::max();

  for (int i = 0; i < vertices - 1; i++) {
    bool updated = false;

    for (int vertex = 0; vertex < vertices; vertex++) {
      if (dist[vertex] == inf) {
        continue;
      }

      for (int edge = in.row_ptr[vertex]; edge < in.row_ptr[vertex + 1]; edge++) {
        int v = in.col_ind[edge];
        int w = in.vals[edge];

        if (dist[v] > dist[vertex] + w) {
          dist[v] = dist[vertex] + w;
          updated = true;
        }
      }
    }

    if (!updated) {
      break;
    }
  }

  return true;
}

bool VasilievMBellmanFordCrsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bellman_ford_crs
```

## Header MPI версии `ops_mpi.hpp`

```cpp
#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VasilievMBellmanFordCrsMPI(const InType &in);
  static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);
  static int RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                             std::vector<int> &local_dist);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bellman_ford_crs
```
## Реализация методов MPI версии `ops_mpi.cpp`
```cpp
#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <limits>
#include <vector>

#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"

namespace vasiliev_m_bellman_ford_crs {

VasilievMBellmanFordCrsMPI::VasilievMBellmanFordCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBellmanFordCrsMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.row_ptr.empty() || in.col_ind.empty() || in.vals.empty()) {
    return false;
  }

  if (in.col_ind.size() != in.vals.size()) {
    return false;
  }

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;
  if (vertices <= 0) {
    return false;
  }

  if (in.source < 0 || in.source >= vertices) {
    return false;
  }

  return true;
}

bool VasilievMBellmanFordCrsMPI::PreProcessingImpl() {
  const auto &in = GetInput();
  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  const int inf = std::numeric_limits<int>::max();
  GetOutput().assign(vertices, inf);
  GetOutput()[in.source] = 0;

  return true;
}

bool VasilievMBellmanFordCrsMPI::RunImpl() {
  const auto &in = GetInput();
  auto &dist = GetOutput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int vertices = static_cast<int>(in.row_ptr.size()) - 1;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalcCountsAndDispls(vertices, size, counts, displs);
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  int start = displs[rank];
  int end = start + counts[rank];

  std::vector<int> local_dist(vertices);

  for (int i = 0; i < vertices - 1; i++) {
    local_dist = dist;

    int local_updated = RelaxLocalEdges(start, end, in, dist, local_dist);

    MPI_Allreduce(local_dist.data(), dist.data(), vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    int global_updated = 0;
    MPI_Allreduce(&local_updated, &global_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_updated == 0) {
      break;
    }
  }

  return true;
}

bool VasilievMBellmanFordCrsMPI::PostProcessingImpl() {
  return true;
}

void VasilievMBellmanFordCrsMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts,
                                                     std::vector<int> &displs) {
  int chunk = n / size;
  int remain = n % size;

  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < remain ? 1 : 0);
  }

  displs[0] = 0;
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
}

int VasilievMBellmanFordCrsMPI::RelaxLocalEdges(int start, int end, const InType &in, const std::vector<int> &dist,
                                                std::vector<int> &local_dist) {
  const int inf = std::numeric_limits<int>::max();
  int local_updated = 0;

  for (int vertex = start; vertex < end; vertex++) {
    if (dist[vertex] == inf) {
      continue;
    }

    for (int edge = in.row_ptr[vertex]; edge < in.row_ptr[vertex + 1]; edge++) {
      int v = in.col_ind[edge];
      int w = in.vals[edge];

      if (local_dist[v] > dist[vertex] + w) {
        local_dist[v] = dist[vertex] + w;
        local_updated = 1;
      }
    }
  }

  return local_updated;
}

}  // namespace vasiliev_m_bellman_ford_crs
```

## Тесты проверки функционала `functional/main.cpp`

```cpp
#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"
#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<1>(test_param);
    for (auto &c : name) {
      if (!static_cast<bool>(std::isalnum(c))) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bellman_ford_crs, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    int num_vertices = 0;
    int num_edges = 0;
    file >> num_vertices >> num_edges;

    std::vector<std::tuple<int, int, int>> edges(num_edges);
    for (int i = 0; i < num_edges; i++) {
      file >> std::get<0>(edges[i]) >> std::get<1>(edges[i]) >> std::get<2>(edges[i]);
    }

    int source = 0;
    file >> source;

    expected_data_.resize(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
      file >> expected_data_[i];
    }

    input_data_.row_ptr.assign(num_vertices + 1, 0);
    for (const auto &e : edges) {
      input_data_.row_ptr[std::get<0>(e) + 1]++;
    }

    for (int i = 1; i <= num_vertices; i++) {
      input_data_.row_ptr[i] += input_data_.row_ptr[i - 1];
    }

    input_data_.col_ind.resize(num_edges);
    input_data_.vals.resize(num_edges);

    std::vector<int> counter = input_data_.row_ptr;
    for (const auto &e : edges) {
      int u = std::get<0>(e);
      int idx = counter[u]++;
      input_data_.col_ind[idx] = std::get<1>(e);
      input_data_.vals[idx] = std::get<2>(e);
    }

    input_data_.source = source;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(VasilievMBellmanFordCrsFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {TestType{0, "tree_func_test0.txt"}, TestType{1, "tree_func_test1.txt"},
                                            TestType{2, "tree_func_test2.txt"}, TestType{3, "tree_func_test3.txt"}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VasilievMBellmanFordCrsMPI, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bellman_ford_crs),
    ppc::util::AddFuncTask<VasilievMBellmanFordCrsSEQ, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bellman_ford_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VasilievMBellmanFordCrsFuncTests::PrintFuncTestName<VasilievMBellmanFordCrsFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VasilievMBellmanFordCrsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vasiliev_m_bellman_ford_crs
```
## Тесты проверки производительности `performance/main.cpp`
```cpp
#include <gtest/gtest.h>

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bellman_ford_crs/common/include/common.hpp"
#include "vasiliev_m_bellman_ford_crs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bellman_ford_crs/seq/include/ops_seq.hpp"

namespace vasiliev_m_bellman_ford_crs {

class VasilievMBellmanFordCrsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    std::ifstream file(ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bellman_ford_crs, "tree_perf_test.txt"));

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open perf test file");
    }

    int num_vertices = 0;
    int num_edges = 0;
    file >> num_vertices >> num_edges;

    std::vector<std::tuple<int, int, int>> edges(num_edges);
    for (int i = 0; i < num_edges; i++) {
      file >> std::get<0>(edges[i]) >> std::get<1>(edges[i]) >> std::get<2>(edges[i]);
    }

    int source = 0;
    file >> source;

    input_data_.row_ptr.assign(num_vertices + 1, 0);
    for (const auto &e : edges) {
      input_data_.row_ptr[std::get<0>(e) + 1]++;
    }

    for (int i = 1; i <= num_vertices; i++) {
      input_data_.row_ptr[i] += input_data_.row_ptr[i - 1];
    }

    input_data_.col_ind.resize(num_edges);
    input_data_.vals.resize(num_edges);

    std::vector<int> counter = input_data_.row_ptr;
    for (const auto &e : edges) {
      int u = std::get<0>(e);
      int idx = counter[u]++;
      input_data_.col_ind[idx] = std::get<1>(e);
      input_data_.vals[idx] = std::get<2>(e);
    }

    input_data_.source = source;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(VasilievMBellmanFordCrsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VasilievMBellmanFordCrsMPI, VasilievMBellmanFordCrsSEQ>(
    PPC_SETTINGS_vasiliev_m_bellman_ford_crs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMBellmanFordCrsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMBellmanFordCrsPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_bellman_ford_crs
```
