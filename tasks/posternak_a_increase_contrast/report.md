# Повышение контраста

- Студент: Постернак Алексей Николаевич, группа 3823Б1ПР2
- Технологии: SEQ | MPI
- Вариант: 23

## 1. Введение

Повышение контраста изображения применяется во многих сферах деятельности человека, такие как медицина (для визуализации снимков), навигация (выделение объектов, запечатленных спутником), машинное обучение (для изучения изображений) и тд.

В некоторых случаях, нужно как можно быстрее повышать контраст изображения (например, для камеры, которая анализирует картинку в реальном времени).

**Цель:**
Разработать параллельную (MPI) реализацию алгоритма повышения контрастности и сравнить производительность с последовательным (SEQ) алгоритмом на различных объемах данных.

## 2. Постановка задачи

**Задача:**
Требуется повысить контраст изображения, которое будет представлено в виде одномерного массива. 

**Ограничения:**
- Изображение не может быть пустым
- Изображение должно успешно загружатся
- Изображение может быть представлено в виде одномерного массива со значениями от `0` до `255`
- Результаты последовательного и параллельного алгоритма должны быть идентичными

## 3. Базывый алгоритм (последовательный)

**Входные данные:**
Исходный одномерный массив, представленный в виде вектора, с переменными типа `unsigned char` (`std::vector<unsigned char> &input`).

**Выходные данные:**
Обработанный одномерный массив, представленный в виде вектора, с переменными типа `unsigned char` (`std::vector<unsigned char> &output`).

**Реализация алгоритма:**

Используем алгоритм линейного растяжения гистограммы:

1. Находим самый темный `min_val` и самый светлый `max_val` пиксели во всём изображении.
2. Если все пиксели одинаковые - заполняем изображение серым цветом (значение пикселя - `128`).
3. Вычисляем масштабный коэффициент по формуле `scale = 255 / (max_val - min_val)`.
4. Преобразовываем каждый пиксель по формуле `new_value = round((old_value - min_val) * 255 / (max_val - min_val))` с округлением результата до целого числа (для приведения полученного значения к типу `unsigned char`).

Полная реализация алгоритма находится в Приложении.

## 4. Схема распараллеливания

**Топология:**
В представленном алгоритме используется декартова топология.

**Распределение данных и параллельные вычисления:**

- Все процессы получают свой ранг и общее количество процессов через `MPI_Comm_rank` и `MPI_Comm_size`
- Нулевой процесс определяет общий размер данных изображения и рассылает его всем процессам через `MPI_Bcast`
- Вычисляется размер локальной части для каждого процесса
- Нулевой процесс распределяет данные с помощью `MPI_Scatterv`:
  - Данные разбиваются на равные части (остаток достается нулевому процессу)
  - Создаются массивы `counts` и `step`, описывающие размер и смещение для каждого процесса
  - Каждый процесс получает свою часть данных
- С помощью `MPI_Allreduce` с операциями `MPI_MIN` и `MPI_MAX` определяются максимум и минимум всего изображения:
```cpp
MPI_Allreduce(&local_min, data_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
MPI_Allreduce(&local_max, data_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
```
- Каждый процесс независимо обрабатывает свою часть изображения:
  - Если `data_min == data_max`, заполняет свою часть серым цветом (`128`)
  - Иначе применяет формулу линейного растяжения гистограммы:
    - Вычислят масштабный коэффициент по формуле `scale = 255 / (max_val - min_val)`.
    - Преобразовывает каждый пиксель по формуле `new_value = round((old_value - min_val) * 255 / (max_val - min_val))` с округлением результата до целого числа (для приведения полученного значения к типу `unsigned char`).
  - Все процессы собирают обработанные части в единое изображение и рассылают результат каждому процессу с помощью `MPI_Allgatherv`:
  ```cpp
  MPI_Allgatherv(local_output.data(), static_cast<int>(local_output.size()), MPI_UNSIGNED_CHAR, GetOutput().data(),
                 counts.data(), step.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
  ```

Полная реализация алгоритма находится в Приложении.

## 5. Детали реализации

**Структура проекта:**
```
- posternak_a_increase_contrast                 // корень проекта
    - common/include/common.hpp                 // определение типов входных и выходных данных
    - mpi                                       // реализация параллельного алгоритма
        - include/ops_mpi.hpp                   // объявление функций
        - src/ops_mpi.cpp                       // реализация функций
    - seq                                       // реализация последовательного алгоритма
        - include/ops_mpi.hpp                   // объявление функций
        - src/ops_mpi.cpp                       // реализация функций
    - test                                      // тестирование алгоритмов mpi и seq
        - functional/main.cpp                   // функциональные тесты
        - perfomance/main.cpp                   // тесты на производительность
    - info.json                                 // информация о студенте
    - report.md                                 // отчет
    - settings.json                             // настройки проекта
```

**Ключевые классы:**
- `PosternakAIncreaseContrastSEQ` - последовательная реализация алгоритма
- `PosternakAIncreaseContrastMPI` - параллельная реализация алгоритма

**Ключевые функции:**
- `ValidationImpl()` - проверка входных данных
- `PreProcessingImpl()` - предварительные вычисления  
- `RunImpl()` - реализация SEQ/MPI алгоритма
- `PostProcessingImpl()` - завершающая обработка

**Частные случаи:**
Предварительное условие - изображение должно быть непустым и успешно загруженым.

## 6. Экспериментальное окружение

**Аппаратное обеспечение:**
- Процессор: Intel Core i7-11800H @ 2.30GHz
- Ядра: 16 шт.
- ОЗУ: 16 ГБ
- ОС: Kubuntu 25.10

**Программный инструментарий:**
- Компилятор: g++ 15.2.0
- Тип сборки: Release
- Стандарт C++: C++23
- MPI: Open MPI 5.0.8

**Тестовое окружение**
```bash
PPC_NUM_PROC=1,2,4
```

## 7. Результаты

### 7.1 Корректность

Все функциональные тесты были успешно пройдены:

1. Простое линейное растяжение.
2. Пикели с большим разбросом (большой перепад яркости).
3. Средний диапозон пикселей.
4. Типичное изображение. 
5. Пиксели с полным диапозоном значений (вывод без изменений).

Граничные тесты:

6. Одноцветное изображение.
7. Черно-белое изображение.
8. Черное изображение.
9. Белое изображение.
10. Изображение с одним пикселем.

SEQ и MPI версии выдают идентичные результаты для всех тестовых случаев.

### 7.2 Производительность

**Результаты замера времени выполнения MPI и SEQ алгоритмов для изображений, размером 8192х8192 пикселей:**

| Режим | Количество процессов | Время, мс | Ускорение | Эффективность |
|-------|----------------------|-----------|-----------|---------------|
| SEQ   | 1                    | 50.76     | 1.00      | N/A           |
| MPI   | 1                    | 80.68     | 0.63      | 63%           |
| MPI   | 2                    | 42.57     | 1.19      | 60%           |
| MPI   | 4                    | 24.38     | 2.08      | 52%           |
| MPI   | 8                    | 18.27     | 2.78      | 35%           |

**Формула ускорения:** Ускорение = Время SEQ / Время MPI

**Формула эффективности:** Эффективность = (Ускорение / Количество процессов) × 100%

### 7.3. Анализ результатов

**Лучшее ускорение:** 2.78 на 8 процессах

**Эффективность:** При увеличении числа процессов снижается показатель эффективности

## 8. Выводы

В результате выполнения проекта были разработаны и протестированы последовательный (SEQ) и параллельный (MPI) алгоритма повышения контрастности изображения.

Параллельный алгоритм работает быстрее последовательного на больших данных, однако на малых немного уступает по времени выполнения от последовательного алгоритма из-за обмена данными между процессами.

При увеличении числа ядер для параллельного алгоритма уменьшается эффективность. В основном, это связанно с увеличением объема обмена данными между процессами.

Итого, для малых объемов данных лучше использовать последовательный (SEQ) алгоритм, а для больших объемов данных - параллельный (MPI) алгоритм.

## 9. Литература

1. Документация по курсу: "Параллельное программирование": https://learning-process.github.io/parallel_programming_course/ru/index.html (Оболенский А.А, Нестеров А.Ю)
2. Лекции по курсу "Параллельное программирование". (Сысоев А.В. ННГУ 2025 г.)
3. Документация по MPI: https://www.open-mpi.org/

## Приложение

`ops_seq.cpp:`

```cpp
bool PosternakAIncreaseContrastSEQ::RunImpl() {
  const std::vector<unsigned char> &input = GetInput();
  std::vector<unsigned char> &output = GetOutput();

  // Находим максимаьный и минимальный пиксели в изображении
  unsigned char min_val = *std::ranges::min_element(input);
  unsigned char max_val = *std::ranges::max_element(input);

  // Если изображение в одном цвете - красим его в серый
  if (min_val == max_val) {
    std::ranges::fill(output, 128);
    return true;
  }

  // Вычисляем масштаб
  double scale = 255.0 / static_cast<double>(max_val - min_val);

  // Преобразуем каждый пиксель с округлением результата
  for (size_t i = 0; i < input.size(); ++i) {
    double new_pixel = static_cast<double>(input[i] - min_val) * scale;
    new_pixel = std::round(new_pixel);
    new_pixel = std::max(new_pixel, 0.0);
    new_pixel = std::min(new_pixel, 255.0);

    output[i] = static_cast<unsigned char>(new_pixel);
  }

  return true;
}
```

`ops_mpi.cpp:`

```cpp
bool PosternakAIncreaseContrastMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int data_len = 0;
  if (rank == 0) {
    data_len = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<unsigned char> proc_part = ScatterInputData(rank, size, data_len);

  unsigned char data_min = 0;
  unsigned char data_max = 0;
  FindGlobalMinMax(proc_part, &data_min, &data_max);

  std::vector<unsigned char> local_output = ApplyContrast(proc_part, data_min, data_max);

  int local_size = data_len / size;
  int remainder = data_len % size;
  std::vector<int> counts(size);
  std::vector<int> step(size);
  int start = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    step[i] = start;
    start += counts[i];
  }

  GetOutput().resize(data_len);
  MPI_Allgatherv(local_output.data(), static_cast<int>(local_output.size()), MPI_UNSIGNED_CHAR, GetOutput().data(),
                 counts.data(), step.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

  return true;
}

std::vector<unsigned char> PosternakAIncreaseContrastMPI::ScatterInputData(int rank, int size, int data_len) {
  int local_size = data_len / size;
  int remainder = data_len % size;
  int my_size = local_size + (rank < remainder ? 1 : 0);

  std::vector<int> counts(size);
  std::vector<int> step(size);
  int start = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = local_size + (i < remainder ? 1 : 0);
    step[i] = start;
    start += counts[i];
  }

  std::vector<unsigned char> proc_part(my_size);
  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), counts.data(), step.data(), MPI_UNSIGNED_CHAR, proc_part.data(), my_size,
                 MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR, proc_part.data(), my_size, MPI_UNSIGNED_CHAR, 0,
                 MPI_COMM_WORLD);
  }
  return proc_part;
}

void PosternakAIncreaseContrastMPI::FindGlobalMinMax(const std::vector<unsigned char> &proc_part,
                                                     unsigned char *data_min, unsigned char *data_max) {
  unsigned char local_min = 255;
  unsigned char local_max = 0;
  for (unsigned char pixel : proc_part) {
    local_min = std::min(local_min, pixel);
    local_max = std::max(local_max, pixel);
  }

  MPI_Allreduce(&local_min, data_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, data_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
}

std::vector<unsigned char> PosternakAIncreaseContrastMPI::ApplyContrast(const std::vector<unsigned char> &proc_part,
                                                                        unsigned char data_min,
                                                                        unsigned char data_max) {
  int my_size = static_cast<int>(proc_part.size());
  std::vector<unsigned char> local_output(my_size);

  if (data_min == data_max) {
    std::ranges::fill(local_output, 128);
  } else {
    const double scale = 255.0 / (data_max - data_min);
    for (int i = 0; i < my_size; ++i) {
      double scaled_value = (proc_part[i] - data_min) * scale;
      int new_pixel = static_cast<int>(std::lround(scaled_value));
      if (new_pixel < 0) {
        new_pixel = 0;
      } else if (new_pixel > 255) {
        new_pixel = 255;
      }
      local_output[i] = static_cast<unsigned char>(new_pixel);
    }
  }
  return local_output;
}

```