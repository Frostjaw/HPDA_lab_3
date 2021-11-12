
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__global__ void gpu_matrix_mult(int* a, int* b, int* c, int matrixA_row_count, int matrixB_row_count, int matrixB_col_count)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < matrixB_col_count && row < matrixA_row_count)
    {
        for (int i = 0; i < matrixB_row_count; i++)
        {
            sum += a[row * matrixB_row_count + i] * b[i * matrixB_col_count + col];
        }
        c[row * matrixB_col_count + col] = sum;
    }
}

void GenerateRandomValues(int* matrix, int row_count, int col_count)
{
    for (int i = 0; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++) {
            matrix[i * col_count + j] = rand() % 100;
        }
    }
}

void GenerateZeroValues(int* matrix, int row_count, int col_count)
{
    for (int i = 0; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++) {
            matrix[i * col_count + j] = 0;
        }
    }
}

void DeallocateMemory(int* matrixA, int* matrixB, int* matrixC, int* matrixC2)
{
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixC2);
}

int main()
{
    srand((unsigned)time(0));

    int n, m;

    cout << "Enter n:" << endl;
    cin >> n;

    cout << "Enter m:" << endl;
    cin >> m;
    
    // GPU
    cout << "GPU implementation:" << endl;

    int *matrixA, *matrixB, *matrixC_gpu;
    matrixA = (int*)malloc(n * m * sizeof(int));
    matrixB = (int*)malloc(m * n * sizeof(int));
    matrixC_gpu = (int*)malloc(n * n * sizeof(int));

    GenerateRandomValues(matrixA, n, m);
    GenerateRandomValues(matrixB, m, n);

    int *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc(&d_matrixA, sizeof(int) * n * m);
    cudaMalloc(&d_matrixB, sizeof(int) * m * n);
    cudaMalloc(&d_matrixC, sizeof(int) * n * n);

    cudaMemcpy(d_matrixA, matrixA, sizeof(int) * n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, matrixB, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    int blockSize = 16;

    unsigned int grid_rows = (m + blockSize - 1) / blockSize;
    unsigned int grid_cols = (n + blockSize - 1) / blockSize;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(blockSize, blockSize);

    auto gpu_t1 = high_resolution_clock::now();
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_matrixA, d_matrixB, d_matrixC, n, m, n);
    cudaDeviceSynchronize();
    auto gpu_t2 = high_resolution_clock::now();

    auto gpu_exe_time_ms = duration_cast<milliseconds>(gpu_t2 - gpu_t1);

    cout << "Execution time: " << gpu_exe_time_ms.count() << "ms" << endl;

    cudaMemcpy(matrixC_gpu, d_matrixC, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    //cout << "A:" << endl;
    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < m; j++) {
    //        cout << matrixA[i * m + j] << " ";
    //    }
    //    cout << endl;
    //}

    //cout << "B:" << endl;
    //for (int i = 0; i < m; i++)
    //{
    //    for (int j = 0; j < n; j++) {
    //        cout << matrixB[i * n + j] << " ";
    //    }
    //    cout << endl;
    //}

    //cout << "C:" << endl;
    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < n; j++) {
    //        cout << matrixC_gpu[i * n + j] << " ";
    //    }
    //    cout << endl;
    //}


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // CPU
    cout << "CPU implementation:" << endl;

    int* matrixC_cpu;

    matrixC_cpu = (int*)malloc(n * n * sizeof(int));

    GenerateZeroValues(matrixC_cpu, n, n);

    auto cpu_t1 = high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            for (int j = 0; j < n; j++)
            {
                matrixC_cpu[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
            }
        }
    }
    auto cpu_t2 = high_resolution_clock::now();

    auto cpu_exe_time_ms = duration_cast<milliseconds>(cpu_t2 - cpu_t1);

    cout << "Execution time: " << cpu_exe_time_ms.count() << "ms" << endl;

    //cout << "C:" << endl;
    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < n; j++) {
    //        cout << matrixC_cpu[i * n + j] << " ";
    //    }
    //    cout << endl;
    //}

    // check
    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < n; j++)
    //    {
    //        if (matrixC_gpu[i * n + j] != matrixC_cpu[i * n + j]) 
    //        {
    //            cout << "Error";
    //        }
    //    }
    //}

    DeallocateMemory(matrixA, matrixB, matrixC_gpu, matrixC_cpu);

    return 0;
}