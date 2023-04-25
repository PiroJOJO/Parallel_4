#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
//Функция, которая высчитывает разницу между двумя массивами 
__global__ void my_sub(double* arr, double* new_arr, double* c, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//Индекс для обращения к элементу массива 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;//Высчитывается из логики - Номер блока на размер блока(кол-во поток) плюс номер потока
    if((i > 0 && i < n-1) && (j > 0 && j < n-1))
    {
        c[i*n + j] = fabs(new_arr[i*n + j] - arr[i*n + j]);//Обращение по индексу по логике преобразования матрицы в одномерный массив
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

__global__ void fill(double* arr, double* new_arr, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    arr[0] = new_arr[0] = 10;
    arr[n - 1]= new_arr[n - 1] = 20;
    arr[n * n - 1] = new_arr[n * n - 1] = 30;
    arr[n * (n - 1)] = new_arr[n * (n - 1)] = 20;
    if(i > 0 && i < n-1)
    {
        arr[i] = new_arr[i] = arr[0] + (10.0 / (n-1)) * i;
        arr[n*(n-1) + i] = new_arr[n*(n-1) + i] = arr[n - 1] + 10.0 / (n-1) * i;
        arr[n*i]= new_arr[n*i] = arr[0] + 10.0 / (n-1) * i;
        arr[n*i + n - 1] = new_arr[n*i + n - 1] = arr[n-1] + 10.0 / (n-1) * i;
    }
} 
void swap(double* &a, double* &b)
{
    double *temp = a;
    a = b;
    b = temp;
}
//Функция для просмотра матрицы
void print_matrix(double* vec, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout<<vec[n*i + j]<<' ';
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
    //cudaSetDevice(0);

    //Объявляем необходимы перменные 
    double* vec = new double[n*n];//Массив для значений на предыдущем шаге
    double* new_vec = new double[n*n];//Массив для значений на текущем шаге
    double* tmp = new double[n*n];//Вспомогаетльный массив для сохранения результата для следующей итерации 
    double* vec_d;
    double* new_vec_d;//Массив для значений на текущем шаге
    double* tmp_d;
    cudaMalloc((void **)&vec_d, sizeof(double)*n*n);
    cudaMalloc((void **)&new_vec_d, sizeof(double)*n*n);
    cudaMalloc((void **)&tmp_d, sizeof(double)*n*n);
    cudaMemcpy(vec_d, vec, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(new_vec_d, new_vec, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_d, tmp, sizeof(double), cudaMemcpyHostToDevice);

    double max_error = error + 1; //Объявление максимальной ошибки 
    size_t it = 0;//Счетчик итераций
    // double stepx = 10/(n-1);
    // cudaMemcpyToSymbol(step, &stepx, sizeof(double));

    //Задаем размер блока и сетки 
    dim3 BLOCK_SIZE = dim3(32, 32);//Размер блока - количество потоков
    dim3 GRID_SIZE = dim3(ceil(n/32.), ceil(n/32.));//Размер сетки - количество блоков
    //Заполнение угловых значений
    //Не забываем, что мы матрицу представляем, как одномерный вектор(вытягиваем ее по по строкам)
    fill<<<GRID_SIZE, BLOCK_SIZE>>>(vec_d, new_vec_d, n);
    cudaMemcpy(vec, vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_vec, new_vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
 
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
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int i = 0; i<n*n/4;i+=2)
    {
        update<<<GRID_SIZE,BLOCK_SIZE, 0, stream>>>(new_vec_d, vec_d, n);
        update<<<GRID_SIZE,BLOCK_SIZE, 0, stream>>>(vec_d, new_vec_d, n);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    while(error < max_error && it < iter)
	{        
        it+=n*n/2;
        cudaGraphLaunch(instance, stream);
        my_sub<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(vec_d, new_vec_d, tmp_d, n);
	    cub::DeviceReduce::Max(store, bytes, tmp_d, max_errorx, n*n);
        cudaMemcpy(&max_error, max_errorx, sizeof(double), cudaMemcpyDeviceToHost);//Обновление ошибки на CPU

    }

    std::cout<<"Error: "<<max_error<<std::endl;
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
    // cudaDeviceSynchronize();
    // cudaMemcpy(vec, vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    // cudaMemcpy(new_vec, new_vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    // print_matrix(vec, n);
    std::cout<<"time: "<<elapsed_ms.count()<<" mcs\n";
    std::cout<<"Iterations: "<<it<<std::endl;
    delete [] vec; 
    delete [] new_vec;
    cudaFree(vec);
    cudaFree(new_vec);
    cudaFree(tmp);
    return 0;  
}