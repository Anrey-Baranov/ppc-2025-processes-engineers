# Введение
В задании 1 предлагается реализовать свою версию функции MPI_Bcast с построением дерева процессов.

# Постановка задачи
Необходимо реализовать функцию широковещательной рассылки с построением дерева процессов.
Ограничения:
* Функция должна уметь передавать int, float, double;
* Должна быть возможность выбора корневого процесса.
# Описание алгоритма
Сначала процессу присваивается внутренний номер, у корневого процесса внутренний корень равен 0.
Далее высчитываются номера родителя и потомков процесса. Для этого алгоритм использует логарифмические и степенные функции. По сути так строится дерево процессов. В функции MPI_Recv, MPI_Send передаются реальные номера процессов.
# Схема распараллеливания
*Корневой процесс считывает данные и рассылает их другим процессам. Для этого все процессы вызывают учебную реализацию MPI_Bcast.
*Внутри функции процессами присваивается локальный индекс. Корневой процесс получает локальный индекс 0.
*Корневой процесс отправляет данные(MPI_Send) процессу с локальным номером 1.
*Остальные процессы вычисляют локальный индекс своего родителя и ждут от него посылки (MPI_Recv).
*Корневой процесс после первой посылки 1-му процессу и все остальные процессы вычисляют своих потомков и отправляют им данные.
Во всех MPI_Recv, MPI_Send используются реальные номера процессов. Вычисление локального номера процесса, его потомков и родителя можно назвать построением дерева процессов.

Полный код реализации в приложении.
# Экспериментальные результаты

## Тестовые данные
Для анализа производительности функции использовались векторы на миллион элементов, три массива в сумме три миллиона элементов.


| Название теста | Количество процессов | Среднее время |
|----------------|---------------------|---------------|
| golovanov_d_bcast_mpi_enabled:pipeline | 1 | 0.0023340644 |
| golovanov_d_bcast_mpi_enabled:task_run | 1 | 0.0024370072 |
| golovanov_d_bcast_mpi_enabled:pipeline | 2 | 0.0087240861 |
| golovanov_d_bcast_mpi_enabled:task_run | 2 | 0.0057507450 |
| golovanov_d_bcast_mpi_enabled:pipeline | 4 | 0.0107761561 |
| golovanov_d_bcast_mpi_enabled:task_run | 4 | 0.0109302539 |
| golovanov_d_bcast_mpi_enabled:pipeline | 6 | 0.0216505695 |
| golovanov_d_bcast_mpi_enabled:task_run | 6 | 0.0267307211 |

*Среднее время написано для 10 выполнений тестов.
*Сравнение с последовательной реализаций не проводилось, потому что там заглушка.

## Окружение
| Параметр       | Значение                                          |
| -------------- | ------------------------------------------------- |
| **OS**         | Windows 11 Pro 25H2                               |
| **CPU**        | AMD Ryzen 5 5600X (6 cores, 12 threads, 3.70 GHz) |
| **RAM**        | 16 GB DDR4 3400 МГц                               |
| **Компилятор** | MSVC 14.43.34808                                  |
| **Версия MPI** | Microsoft MPI 10.1.12498.52                       |


# Результаты

Требуемая функция успешно реализована и проходит функциональные и производительные тесты. Однако сравнение с оригинальной функцией не проводилось. 



# Источники
1. https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/ - статья, на которую я опирался.
# Приложение
## Код параллельной версии

```
static int GolovanovDBcastMPI::MyBcast(void *buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm comm){
  int real_rank = 0;
  MPI_Comm_rank(comm, &real_rank);
  int world_size;
  MPI_Comm_size(comm, &world_size);
  int local_rank = (real_rank - root + world_size) % world_size;
  int rank_lvl = 0;
  //корень отправляет первому
  if(local_rank == 0)
  {
    if(world_size > 1)
    {
      rank_lvl = 1;
      int local_child = 1; 
      int real_child = (root + local_child) % world_size;
      MPI_Send(buffer, count, datatype, real_child, 0, comm);
    }
  }
  //не-корень получает впервые
  else 
  {
    rank_lvl = static_cast<int>(floor(log2(local_rank))) + 1;
    int parent_offset = static_cast<int>(pow(2, rank_lvl - 1));
    int local_parent = local_rank - parent_offset;
    int real_parent = (root + local_parent) % world_size;
    MPI_Recv(buffer, count, datatype, real_parent, 0, comm, MPI_STATUS_IGNORE);
  }
  //расслыка
  int local_child = local_rank + static_cast<int>(pow(2, rank_lvl));
  while(local_child < world_size)
  {
    int real_child = (root + local_child) % world_size;
    MPI_Send(buffer, count, datatype, real_child, 0, comm);
    rank_lvl++;
    local_child = local_rank + static_cast<int>(pow(2, rank_lvl));
  }
  return MPI_SUCCESS;
}

```
