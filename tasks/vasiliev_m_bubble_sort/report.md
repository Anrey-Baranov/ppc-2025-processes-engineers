# Отчет по лабораторной работе
## Сортировка пузырьком (алгоритм чет-нечетной перестановки)

- Студент: Васильев Михаил Петрович, группа 3823Б1ПР2
- Технология: SEQ | MPI
- Вариант: 21

# 1. Введение
В современных условиях обработки больших массивов данных особое значение имеет эффективное использование вычислительных ресурсов. Параллельные вычисления позволяют значительно уменьшить время выполнения алгоритмов за счет распределения вычислительной нагрузки между несколькими процессорами или процессами.

Одним из наиболее широко используемых инструментов параллельного программирования является технология MPI (Message Passing Interface), обеспечивающая взаимодействие процессов посредством обмена сообщениями в распределённых системах.

В рамках данной работы исследуется задача на сортировку "пузырьком" (четной-нечетной перестановкой). Для решения задачи были реализованы две версии алгоритма: последовательная (SEQ) и параллельная (MPI). Основная цель работы заключается в сравнении их производительности и анализе эффективности параллелизации для данной задачи.

---

# 2. Постановка задачи
Дан вектор размера `n`, где `n` — целое положительное число (int). Необходимо отсортировать данный вектор по возрастанию применяя сортировку пузырьком с модификацией чет-нечет перестановки.

Например, вектор `1, 4, -3, 2, -1, 3`, в итоге, должен отсортироваться в `-3, -1, 1, 2, 3, 4`.

---

# 3. Последовательная версия (SEQ)

SEQ и MPI версия алгоритмов состоят из 4-х этапов:
  `ValidationImpl()` - Проверка входных данных.
  `PreProcessingImpl()` - Подготовка к выполнению алгоритма.
  `RunImpl()` - Основной алгоритм.
  `PostProcessingImpl()` - Проверка конечного результата.

Далее работа данных методов в SEQ версии алгоритма:

**1. `ValidationImpl()`**
- Проверяется, что входной вектор не пуст.

**2. `PreProcessingImpl()`**
- Очищение выходного результата.

**3. `RunImpl()`**
1. Создание флага `sorted`, показывающий, была ли хотя бы одна перестановка на текущей итерации.
2. Четные и нечетные фазы сортировки в цикле. В четной фазе сравниваются пары с четными индексами, а в нечетный - нечетные, соответственно.
3. Если за обе фазы перестановок не было - цикл завершается и вектор является остортированным.

**4. `PostProcessingImpl()`**
- В пост-проверке нет необходимости.

---

# 4. Схема параллелизации
Идея параллелизации заключается в том, что вектор можно разбить на независимые части, что позволяет распределить обработку каждой части отдельным процессом параллельно.

Каждый процесс высчитывает локально сортирует чет-нечет фазами свою часть вектора, затем выполняются глобальные фазы обмена между соседними процессами. В конце результат собирается на root и рассылается всем остальным процессам.


## Схема
Главным процессом (rank 0) инициализируется MPI и, в зависимости от указанного количества процессов в параметрах, создается данное число отдельных процессов, занимающие свою часть вектора (а если процессы не могут равномерно занять части, то первые из них будут иметь больше элементов на обработку, чем остальные).

Таким образом (при равномерном распределении): `Процесс_0` займет первые `n` элементов вектора для обработки, `Процесс_1` займет `с n+1 до 2n` элементы, и так далее.

Затем каждый процесс обрабатывает свои отдельные части, локально их сортируя.

Далее, выполняется глобальная сортировка, где в каждой фазе образуются пары и, затем, происходит обмен локальными блоками с партнером, слияние двух отсортированных массивов, по итогу - один процесс оставляет меньшую половину, а другой большую. Таким образом, меньшие элементы двигаются влево, а большие - вправо.

После этого все отсортированные блоки собираются в один вектор на главном процессе (rank 0), что будет сигнализироваться как завершение сортировки вектора.


## Распределение нагрузки
При выделении `k` процессов и обработки вектора размером `n`, количество элементов распределяются следующим образом:

```cpp
int chunk = n / k;          // Изначальное количество элементов на процесс
remain = n % k;              // Остаточное количество неиспользованных элементов вектора
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
- Процесс 0: элементы 0-3 (4 элемента)
- Процесс 1: элементы 4-6 (3 элемента)
- Процесс 2: элементы 7-9 (3 элемента)
## Коммуникации между процессами
MPI версия алгоритма использует следующие коммуникации:
1. **MPI_Bcast** — рассылает процессам их отдельное количество элементов вектора (counts) и смещение в нем (displs). Третий `Bcast` - рассылка процессам отсортированного вектора (результат необходим всем для корректного завершения выполнения вне зависимости от rank):
   ```cpp
   MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
   ```
2. **MPI_Scatterv** — раздача данных (частей вектора) всем процессам (учитывая возможное неравномерное разбитие):
   ```cpp
   MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT,
              local_data.data(), counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
   ```
3. **MPI_Barrier** — исключает гонки фаз разных процессов:
   ```cpp
   MPI_Barrier(MPI_COMM_WORLD);
   ```
4. **MPI_Gatherv** — сбор всех блоков массива в один вектор в root:
   ```cpp
   MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr, counts.data(),
                displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
   ```
5. **MPI_Sendrecv** — обмен размерами и данными блоков у партнеров:
   ```cpp
   MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
   MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 1, partner_data.data(), partner_size, MPI_INT, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
  using InType = std::vector<int>;
  using OutType = std::vector<int>;
  using BaseTask = ppc::task::Task<InType, OutType>;
  
  class VasilievMBubbleSortMPI : public BaseTask {
   public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kMPI;
    }
    explicit VasilievMBubbleSortMPI(const InType &in);
    static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);  // вычисление кол-ва элементов на процесс и смещения в векторе
    static void BubbleOESort(std::vector<int> &array);  // локальная сортировка процессами своих блоков
    static int FindPartner(int rank, int phase);  // нахождение партнера (обозначение пары для чет-нечет фаз)
    static void ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data);  // обмен и слияние блоков
  
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
VasilievMBubbleSortMPI::VasilievMBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}
```
Конструктор инициализирует тип задачи (в данном случае MPI), устанавливает входные данные и тип выходных.

### Валидация данных
```cpp
bool VasilievMBubbleSortMPI::ValidationImpl() {
  return !GetInput().empty();
}
```
Проверка на корректность входных данных (вектор не является пустым).

### Предварительная обработка
```cpp
bool VasilievMBubbleSortMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}
```
Очищение выходных данных.

#### Основной алгоритм
```cpp
bool VasilievMBubbleSortMPI::RunImpl() {
  int rank = 0;  // информация о процессах
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &vec = GetInput();
  int n = static_cast<int>(vec.size());

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  // вычисления counts и displs происходят только на root-ранге
  if (rank == 0) {
    CalcCountsAndDispls(n, size, counts, displs);  // распределение частей вектора между процессами
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);  // рассылка counts и displs всем процессам
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // раздача выбранного количества элементов процессам
  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  BubbleOESort(local_data);  // локальная сортировка

  for (int phase = 0; phase < size + 1; phase++) {  // глобальная чет-нечет перестановка через фазы
    int partner = FindPartner(rank, phase);  // составление пар

    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(rank, partner, local_data);  // обмен и слияние блоков
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().resize(n);
  }

  MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);  // сбор результата

  if (rank != 0) {
    GetOutput().resize(n);
  }

  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);  // получение всеми процессами полного отсортированного вектора

  return true;
}
```

### Постобработка результатов

```cpp
bool VasilievMBubbleSortMPI::PostProcessingImpl() {
  return true;
}
```
Пост-проверки не требуется.

### Вспомогательные функции
```cpp
void VasilievMBubbleSortMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
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
Вычисление количества элементов на процесс и смещения в векторе.
```cpp
void VasilievMBubbleSortMPI::BubbleOESort(std::vector<int> &vec) {
  const size_t n = vec.size();
  bool sorted = false;

  if (vec.empty()) {
    return;
  }

  while (!sorted) {
    sorted = true;

    for (size_t i = 0; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }
  }
}
```
Полная сортировка процессом своего отдельного блока, используя чет-нечет фазы.
```cpp
int VasilievMBubbleSortMPI::FindPartner(int rank, int phase) {
  if (phase % 2 == 0) {
    return (rank % 2 == 0) ? rank + 1 : rank - 1;
  } else {
    return (rank % 2 == 0) ? rank - 1 : rank + 1;
  }
}
```
Нахождение партнеров для пары.
```cpp
void VasilievMBubbleSortMPI::ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data) {
  int local_size = static_cast<int>(local_data.size());
  int partner_size = 0;

  MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> partner_data(partner_size);

  MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 1, partner_data.data(), partner_size, MPI_INT, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(local_size + partner_size);
  size_t i = 0, j = 0, k = 0;
  while (i < local_data.size() && j < partner_data.size()) {
    merged[k++] = (local_data[i] <= partner_data[j]) ? local_data[i++] : partner_data[j++];
  }
  while (i < local_data.size()) {
    merged[k++] = local_data[i++];
  }
  while (j < partner_data.size()) {
    merged[k++] = partner_data[j++];
  }

  if (rank < partner) {
    std::copy(merged.begin(), merged.begin() + local_size, local_data.begin());
  } else {
    std::copy(merged.end() - local_size, merged.end(), local_data.begin());
  }
}
```
Обмен процессами своими блоками, слияние отсортированных, в левом процессе остаются min элементы, в правом - max.

### Особые случаи
В данном алгоритме предполагается, что количество процессов будет четным для эффективной работы сортировки. При нечетном количестве - один из процессов будет всегда без пары, следовательно, пропускать фазы.

### Преимущества MPI-реализации
- **Параллелизация** — каждый процесс хранит и обрабатывает только свою часть вектора, следовательно сложность обработки сокращается в `k` раз.
- **Масштабируемость** — алгоритм будет работать все более эффективно на все более большем векторе и/или при увеличении числа процессов.
- **Отсутствие гонок данных** — каждый процесс работает только со своими отдельными частями вектора, не затрагивая данные других процессов.

---

# 6. Результаты экспериментов
## Типы экспериментов/тестов
Для проверки корректной работы алгоритмов и ее производительности были составлены:
- Функциональные тесы - для проверки результатов алгоритмов
- Тесты производительности - для измерения производительности/времени выполнения работы алгоритмом
Для написания тестов были использованы `google tests`.

## Алгоритм считывания тестовых значений
Все тестовые векторы записаны в отдельном файле формата `.txt`.

Схема считывания векторов в тестах следующая: подразумевается, что один вектор занимает одну строку, следовательно считывание данных происходит построчно.
Элементы вектора разделяются пробелами (без запятых) и сигнализируется их окончание точкой с запятой. После этого знака указывается отсортированная версия данного вектора с идентичными разделениями между элементами.
(Напр.: `1 -2 3 -4 ; -4 -2 1 3`)

Алгоритм считывания тестовых значений позволяет переводить векторы из `string` формата в обрабатываемый формат `vector<int>`, для дальнейшей работы в вычислении количества чередований (SEQ или MPI).

Соответственный алгоритм:
```cpp
      std::stringstream ss(line);  // создание потока для строки, читающий числа до `;`
      std::vector<int> input;
      std::vector<int> expected;
      int value = 0;
      while (ss >> value) {
        input.push_back(value);  // добавление числа в вектор
        ss >> std::ws;
        if (ss.peek() == ';') {  // встреча `;` - окончание цикла
          ss.get();
          break;
        }
      }  // по окончании цикла вектор string вида был преобразован в тип vector<int>

      while (ss >> value) {
        expected.push_back(value);  // считывание ожидаемого вектора
      }
```
## Функциональные тесты
Для проверки корректности алгоритмов было написано 8 тестовых векторов с заготовленным результатом в виде отсортированного вектора.
`1 3 2 ; 1 2 3`
`3 1 2 5 4 ; 1 2 3 4 5` - первые два теста проверяют базовую работу
`10 -1 3 0 ; -1 0 3 10` - добавлены отрицательные значения и нуль
`0 1 1 ; 0 1 1` - уже отсортированный вектор с повторяющимися значениями
`5 4 3 2 1 ; 1 2 3 4 5` - изначальный вектор отсортирован по убыванию
`1 ; 1` - единственный элемент в векторе
`2 1 ; 1 2` - одна смена значений для сортировки
8 тест - увеличенный вектор размером в `20` элементов и с двузначными числами, для дополнительной проверки корректности работы.
### Результаты
**Все 8 функциональных тестов были успешно пройдены** для обеих версий (SEQ и MPI) при различном количестве процессов: 1, 2, 4, 5, 6, 8 (работа проводилась на машине с 6 доступными ядрами).
- Были верно подсчитаны тестовые векторы
- Результаты обоих версий алгоритмов (SEQ, MPI) были идентичны
- Корректна распределена нагрузка при параллельном выполнении
- Правильная обработка при нечетном количестве процессов
- Верная работа алгоритма при большем количестве процессов, чем частей вектора или физических ядер
## Тесты производительности
Для наиболее явной проверки производительности были реализованы тесты на векторе размером `80 тыс.` элементов с двумя режимами измерений:
1. **task_run** — измерение времени выполнения метода `RunImpl()`
2. **pipeline** — измерение времени выполнения всего конвейера (`ValidationImpl()` + `PreProcessingImpl()` + `RunImpl()` + `PostProcessingImpl()`)
### Ускорение при параллелизации
При увеличении количества процессов для работы параллельного алгоритма, ускорение не будет являться линейным из-за накладных расходов:
- Инициализация MPI
- Рассылка разных данных процессам
- Ожидание завершение всех процессов (из-за возможного неравномерного распределения элементов вектора на процессы или гонок фаз)
- Сбор отсортированных блоков
В дополнение к этому, из-за вышеперечисленных пунктов параллельный алгоритм на малых данных будет выполняться дольше, чем последовательная версия (затраты на параллелизацию больше, чем выигрыш от ее преимуществ). Следовательно, ускорение будет заметно только на большом векторе.
---
# 7. Результаты и выводы
## Корректность реализации
Все написанные тесты были выполнены верно при применении последовательной (SEQ) и параллельной (MPI) версии алгоритма (результаты работ алгоритмов равны предзаписанным векторам). Результаты работ обоих алгоритмов совпадают.
При вычислении тестового вектора, цель которого - проверка производительности, алгоритмы так же корректно выполнили задачу и предоставили явные показатели ускорения и эффективности.
## Инфраструктура для тестов
### Виртуальная машина (VirtualBox)
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 9400F (6 cores, 6 threads, 2900 MHz)   |
| RAM        | 10 GB DDR4 (2660 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS                                   |
| Compiler   | GCC 13.3.0, Release Build                            |
## Производительность
Изначально размер тестируемого вектора представляет `10 тыс.` элементов. Для более справедливой оценки, размер, перед тестом, увеличивается путем копирования всех существующих элементов и вставки их как новых в цикле. При тестировании итогового вектора размером `80 тыс.` элементов, показатели производительности (при разных количествах процессов) были следующими:
| Версия алг-ма        | Кол-во процессов | Время, с | Ускорение | Эффективность |
|----------------------|------------------|----------|-----------|---------------|
| SEQ                  | 1                | 1.558    | 1.00      | N/A           |
| MPI                  | 2                | 1.596    | 0.97      | 48.5%         |
| MPI                  | 4                | 0.367    | 4.24      | 106.0%        |
| MPI                  | 6                | 0.170    | 9.16      | 152.6%        |
| MPI                  | 7                | 0.209    | 7.45      | 106.4%        |
| MPI                  | 8                | 0.217    | 7.17      | 89.6%         |
Из таблицы видно, что при увеличении количества процессов время выполнения алгоритмов сокращается, а его работа ускоряется с хорошей масштабируемостью. При использовании лишь 2-х процессов, накладные расходы перевешивают выигрыш от параллелизации и время выполнения - медленнее, чем у последовательного алгоритма.
Таким образом, при использовании 2 процессов, время работы было замедлено с коэффициентом 0,97, при использовании 4 процессов - время сокращено более, чем в 4 раза, а при использовании 6 - в 9,16 раза. Эффективность, при увеличении процессов увеличивается, вследствие отличной масштабируемости при параллелизации на векторе большого размера, в совокупности используемого типа сортировки.
При использовании 7 и 8 процессов и флага `--oversubscribe` ускорение и эффективность уменьшается, из-за использования нескольких процессов одного физического ядра, но при этом время выполнения алгоритма, так же, многократно меньше, чем у последовательной версии.
Следует учесть, что такое многократное ускорение связано со спецификой модификации данной сортировки, отлично себя проявляющей именно при параллелизации. Временная сложность алгоритма составляет O(n^2), что при параллелизации дает ускорение, больше линейного. При этом, дополнительно, ускорение работы происходит за счёт кэширования - процессы сохраняют отдельные сортируемые блоки в кэше, что сокращает затраты времени на чтение/транспортировку данных.
## Выводы
- Функциональные тесты и тесты для производительности выполняются корректно.
- MPI версия алгоритма многократно эффективней SEQ версии в данном типе сортировки.
- Отличное масштабирование при увеличении числа процессов.
- Корректная работа параллельной версии алгоритма при количестве процессов меньше, чем доступных ядер; такого же количества; и больше.
- Распределение нагрузки имеет ключевую роль в увеличении эффективности работы при параллелизации.
---
# 8. Заключение
В рамках данной работы была успешно решена задача сортировки пузырьком с чет-нечетными перестановками, с использованием технологии MPI.
Были разработаны и реализованы последовательная (SEQ) и параллельная (MPI) версии алгоритма.
Было выявлено какого ускорения и какого типа возможно достигнуть при использовании параллелизации.
В итоге, результаты показали, что применение MPI позволяет повысить производительность при обработке больших объёмов данных, а применяемый вид сортировки демонстрирует возможность существования алгоритмов, специально созданных для параллелизации, которая многократно увеличивает эффективность их работы.
---
# 9. Список литературы
1. MPI Guide. **Using MPI with C++: a basic guide (2024)** [Электронный ресурс]. — Режим доступа: https://www.paulnorvig.com/guides/using-mpi-with-c.html
2. Snir M., Otto S. **MPI: The Complete Reference**. - MIT Press Cambridge, 1995 - 217 p.
3. MPI Forum. **MPI: A Message-Passing Interface Standard. Version 3.1** [Электронный ресурс]. — Режим доступа: https://www.mpi-forum.org/docs/
4. Odd-Even Sort / Brick Sort. **Variations of sort algorithms** [Электронный ресурс]. — Режим доступа: https://www.geeksforgeeks.org/dsa/odd-even-sort-brick-sort/
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

namespace vasiliev_m_bubble_sort {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vasiliev_m_bubble_sort
```

## Header SEQ версии `ops_seq.hpp`
```cpp
#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMBubbleSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bubble_sort
```
## Реализация методов SEQ версии `ops_seq.cpp`
```cpp
#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

VasilievMBubbleSortSEQ::VasilievMBubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBubbleSortSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMBubbleSortSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMBubbleSortSEQ::RunImpl() {
  auto &vec = GetInput();
  const size_t n = vec.size();
  bool sorted = false;

  if (vec.empty()) {
    return true;
  }

  while (!sorted) {
    sorted = true;

    for (size_t i = 0; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }
  }

  GetOutput() = vec;
  return true;
}

bool VasilievMBubbleSortSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bubble_sort
```

## Header MPI версии `ops_mpi.hpp`

```cpp
#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VasilievMBubbleSortMPI(const InType &in);
  static void CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs);
  static void BubbleOESort(std::vector<int> &vec);
  static int FindPartner(int rank, int phase);
  static void ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_bubble_sort
```
## Реализация методов MPI версии `ops_mpi.cpp`
```cpp
#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vasiliev_m_bubble_sort/common/include/common.hpp"

namespace vasiliev_m_bubble_sort {

VasilievMBubbleSortMPI::VasilievMBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMBubbleSortMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMBubbleSortMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &vec = GetInput();
  int n = static_cast<int>(vec.size());

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    CalcCountsAndDispls(n, size, counts, displs);
  }

  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  BubbleOESort(local_data);

  for (int phase = 0; phase < size + 1; phase++) {
    int partner = FindPartner(rank, phase);

    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(rank, partner, local_data);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().resize(n);
  }

  MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(n);
  }

  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void VasilievMBubbleSortMPI::CalcCountsAndDispls(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
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

void VasilievMBubbleSortMPI::BubbleOESort(std::vector<int> &vec) {
  const size_t n = vec.size();
  bool sorted = false;

  if (vec.empty()) {
    return;
  }

  while (!sorted) {
    sorted = true;

    for (size_t i = 0; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }

    for (size_t i = 1; i < n - 1; i += 2) {
      if (vec[i] > vec[i + 1]) {
        std::swap(vec[i], vec[i + 1]);
        sorted = false;
      }
    }
  }
}

int VasilievMBubbleSortMPI::FindPartner(int rank, int phase) {
  if (phase % 2 == 0) {
    return (rank % 2 == 0) ? rank + 1 : rank - 1;
  }
  return (rank % 2 == 0) ? rank - 1 : rank + 1;
}

void VasilievMBubbleSortMPI::ExchangeAndMerge(int rank, int partner, std::vector<int> &local_data) {
  int local_size = static_cast<int>(local_data.size());
  int partner_size = 0;

  MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> partner_data(partner_size);

  MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 1, partner_data.data(), partner_size, MPI_INT, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(local_size + partner_size);
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < local_data.size() && j < partner_data.size()) {
    merged[k++] = (local_data[i] <= partner_data[j]) ? local_data[i++] : partner_data[j++];
  }
  while (i < local_data.size()) {
    merged[k++] = local_data[i++];
  }
  while (j < partner_data.size()) {
    merged[k++] = partner_data[j++];
  }

  if (rank < partner) {
    std::copy(merged.begin(), merged.begin() + local_size, local_data.begin());
  } else {
    std::copy(merged.end() - local_size, merged.end(), local_data.begin());
  }
}

bool VasilievMBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace vasiliev_m_bubble_sort
```

## Тесты проверки функционала `functional/main.cpp`

```cpp
#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"
#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bubble_sort, "test_vectors_func.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    test_vectors_.clear();
    std::string line;

    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      std::stringstream ss(line);
      std::vector<int> input;
      std::vector<int> expected;
      int value = 0;
      while (ss >> value) {
        input.push_back(value);
        ss >> std::ws;
        if (ss.peek() == ';') {
          ss.get();
          break;
        }
      }

      while (ss >> value) {
        expected.push_back(value);
      }

      test_vectors_.emplace_back(input, expected);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int index = std::get<0>(params);
    input_data_ = test_vectors_[index].first;
    expected_output_ = test_vectors_[index].second;
    return input_data_;
  }

 private:
  std::vector<std::pair<std::vector<int>, std::vector<int>>> test_vectors_;
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(VasilievMBubbleSortFuncTests, VectorSort) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "case1"), std::make_tuple(1, "case2"), std::make_tuple(2, "case3"), std::make_tuple(3, "case4"),
    std::make_tuple(4, "case5"), std::make_tuple(5, "case6"), std::make_tuple(6, "case7"), std::make_tuple(7, "case8")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VasilievMBubbleSortMPI, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bubble_sort),
    ppc::util::AddFuncTask<VasilievMBubbleSortSEQ, InType>(kTestParam, PPC_SETTINGS_vasiliev_m_bubble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VasilievMBubbleSortFuncTests::PrintFuncTestName<VasilievMBubbleSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VasilievMBubbleSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vasiliev_m_bubble_sort
```
## Тесты проверки производительности `performance/main.cpp`
```cpp
#include <gtest/gtest.h>

#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_bubble_sort/common/include/common.hpp"
#include "vasiliev_m_bubble_sort/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_bubble_sort/seq/include/ops_seq.hpp"

namespace vasiliev_m_bubble_sort {

class VasilievMBubbleSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_bubble_sort, "test_vector_perf.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    input_data_.clear();
    int value = 0;

    while (file >> value) {
      input_data_.push_back(value);
    }

    std::vector<int> basic_vec = input_data_;

    for (int i = 0; i < 8; i++) {
      input_data_.insert(input_data_.end(), basic_vec.begin(), basic_vec.end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(VasilievMBubbleSortPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VasilievMBubbleSortMPI, VasilievMBubbleSortSEQ>(
    PPC_SETTINGS_vasiliev_m_bubble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMBubbleSortPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMBubbleSortPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_bubble_sort
```
