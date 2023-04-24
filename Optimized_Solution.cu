#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cub/cub.cuh>
//Функция, которая высчитывает разницу между двумя массивами 
__global__ void my_sub(double* arr, double* new_arr, double* c, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//Индекс для обращения к элементу массива 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;//Высчитывается из логики - Номер блока на размер блока(кол-во поток) плюс номер потока
    if((i > 0 && i < n-1) && (j > 0 && j < n-1))
    {
        c[i*n + j] = new_arr[i*n + j] - arr[i*n + j];//Обращение по индексу по логике преобразования матрицы в одномерный массив
    }
} 
//Функция, которая высчитвает средние значения для обновления сетки
__global__ void update(double* arr, double* new_arr, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i > 0 && i < n-1) && (j > 0 && j < n-1))
    {
        new_arr[i*n + j] = 0.25 * (arr[i*n + j - 1] + arr[i*n + j + 1] + arr[(i - 1)*n + j] + arr[(i + 1)*n + j]);
    }
} 
__constant__ double step;
__global__ void fill(double* arr, double* new_arr, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    arr[0] = new_arr[0] = 10;
    arr[n - 1]= new_arr[n - 1] = 20;
    arr[n * n - 1] = new_arr[n * n - 1] = 30;
    arr[n * (n - 1)] = new_arr[n * (n - 1)] = 20;
    if(i > 0 && i < n-1)
    {
        arr[i] = new_arr[i] = arr[i - 1] + step;
        arr[n*(n-1) + i] = new_arr[n*(n-1) + i]= arr[n*(n-1) + i - 1] + step;
        arr[n*i]= new_arr[n*i] = arr[n*(i-1)] + step;
        arr[n*i + n - 1] = new_arr[n*i + n - 1]= arr[n*(i-1) + n - 1] + step;
    }
} 
//Функция для просмотра матрицы
void print_matrix(double* vec, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout<<vec[n*i +j]<<' ';
        }
        std::cout<<std::endl;
    }
}

int main(int argc, char *argv[]) {

    auto begin = std::chrono::steady_clock::now();
    if (argc != 7)
    {
        std::cout<<"Enter a string like this: Accuracy _ iterations _ size _"<<std::endl;
    }

    //Считывание значений с командной строки
    double error = std::stod(argv[2]);//Значение ошибки
    size_t iter = std::stoi(argv[4]);//Количество итераций
    size_t n = std::stoi(argv[6]);//Размер сетки 

    //Объявляем необходимы перменные 
    double* vec;
    double* new_vec;//Массив для значений на текущем шаге
    double* tmp;
    cudaMalloc((void **)&vec, sizeof(double)*n*n);
    cudaMalloc((void **)&new_vec, sizeof(double)*n*n);
    cudaMalloc((void **)&tmp, sizeof(double)*n*n);
    double max_error = error + 1; //Объявление максимальной ошибки 
    size_t it = 0;//Счетчик итераций
    double stepx = 10/(n-1);
    cudaMemcpyToSymbol(step, &stepx, sizeof(double));

    //Задаем размер блока и сетки 
    dim3 BLOCK_SIZE = dim3(32, 32);//Размер блока - количество потоков
    dim3 GRID_SIZE = dim3(ceil(n/32.), ceil(n/32.));//Размер сетки - количество блоков
    //Заполнение угловых значений
    //Не забываем, что мы матрицу представляем, как одномерный вектор(вытягиваем ее по по строкам)

    print_matrix(vec, n);

// #pragma acc data copy(new_vec[0:n*n], vec[0:n*n]) //Переносим занчения на видеокарту
//     {
//         //Заполнение рамок матриц
// #pragma acc parallel loop independent//Создание ядра для распарреллеливания цикла
//         for (size_t i = 1; i < n - 1; ++i) 
//         {
//             vec[i] = steps(n, i, vec[0], vec[n]); //Заполнение значениями для первой строки матрицы
//             vec[n * i] = steps(n, i, vec[n], vec[n * n]);//Заполнение значениями для первого столбца матрицы
//             vec[(n - 1) * n + i] = steps(n, i, vec[n * (n - 1) + 1], vec[n * n]);//Заполнение значениями послендней строки матрицы
//             vec[i * n + n - 1] = steps(n, i, vec[0], vec[n * (n - 1) + 1]);//Заполнение значениями последнего столбца матрицы
//         }
//     }





    //Также инициализируем переменную для расчета максимальной ошибки на cuda
    double* max_errorx;
    cudaMalloc(&max_errorx, sizeof(double));

    //Переменные для работы с библиотекой cub
    void* store = NULL;//Доступное устройство выделения временного хранилища. 
    //При NULL требуемый размер выделения записывается в bytes, и никакая работа не выполняется.
    size_t bytes = 0;//Ссылка на размер в байтах распределения store
    cub::DeviceReduce::Max(store, bytes, vec, max_errorx, n*n);
    // Allocate temporary storage
	cudaMalloc(&store, bytes);
    //Цикл основного алгоритма 
        while(error < max_error && it < iter)
	    {        
            it++;
            update<<<GRID_SIZE,BLOCK_SIZE>>>(vec, new_vec, n);

        if (it % n == 0)
        {
            my_sub<<<GRID_SIZE, BLOCK_SIZE>>>(vec, new_vec, tmp, n);
	        cub::DeviceReduce::Max(store, bytes, tmp, max_errorx, n*n);
            // Allocate temporary storage
	        cudaMalloc(&store, bytes);
            // Run max-reduction
	        cub::DeviceReduce::Max(store, bytes, tmp, max_errorx, n*n);
            cudaMemcpy(&max_error, max_errorx, sizeof(double), cudaMemcpyDeviceToHost);//Обновление ошибки на CPU
        }

         //Обмен между массивами с старыми значенями и с новыми через указатель
        // std::swap(vec, new_vec);

        // acc_attach((void**)&vec);
        // acc_attach((void**)&new_vec);

        }
    

    std::cout<<"Error: "<<max_error<<std::endl;
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
    std::cout<<"time: "<<elapsed_ms.count()<<" mcs\n";
    std::cout<<"Iterations: "<<it<<std::endl;

    //print_matrix(vec, n);
    delete [] vec; 
    delete [] new_vec;
    cudaFree(vec);
    cudaFree(new_vec);
    cudaFree(tmp);
    return 0;  
}