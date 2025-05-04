#include <iostream>
#include <cuda.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ==============================
// CUDA Kernel for Vector Addition
// ==============================
__global__ void vectorAdd(int *A, int *B, int *C, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        C[tid] = A[tid] + B[tid];
    }
}

// ==============================
// CUDA Kernel for Matrix Multiplication
// ==============================
__global__ void matrixMultiply(int *A, int *B, int *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        int sum = 0;
        for (int i = 0; i < N; ++i)
            sum += A[row * N + i] * B[i * K + col];
        C[row * K + col] = sum;
    }
}

// ==============================
// Sequential Vector Addition (CPU)
// ==============================
void sequentialVectorAdd(int *A, int *B, int *C, int size)
{
    for (int i = 0; i < size; ++i)
        C[i] = A[i] + B[i];
}

// ==============================
// Sequential Matrix Multiplication (CPU)
// ==============================
void sequentialMatrixMultiply(int *A, int *B, int *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
        {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];
            C[i * K + j] = sum;
        }
}

// ==============================
// Main Function
// ==============================
int main()
{
    // ========== Vector Addition ==========
    int vecSize = 1 << 20; // 1 million elements

    int *h_A = new int[vecSize];
    int *h_B = new int[vecSize];
    int *h_C_seq = new int[vecSize];
    int *h_C_par = new int[vecSize];

    for (int i = 0; i < vecSize; ++i)
    {
        h_A[i] = i + 1;
        h_B[i] = i + 2;
    }

    int *d_A, *d_B, *d_C;
    size_t vecBytes = vecSize * sizeof(int);
    cudaMalloc(&d_A, vecBytes);
    cudaMalloc(&d_B, vecBytes);
    cudaMalloc(&d_C, vecBytes);

    // Sequential vector addition
    auto start = high_resolution_clock::now();
    sequentialVectorAdd(h_A, h_B, h_C_seq, vecSize);
    auto end = high_resolution_clock::now();
    auto seqVecTime = duration_cast<milliseconds>(end - start).count();

    // Parallel vector addition (GPU)
    cudaMemcpy(d_A, h_A, vecBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vecBytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (vecSize + threads - 1) / threads;

    start = high_resolution_clock::now();
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, vecSize);
    cudaMemcpy(h_C_par, d_C, vecBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    auto parVecTime = duration_cast<milliseconds>(end - start).count();

    cout << "\n===== Vector Addition Results =====\n";
    cout << "Sequential Time: " << seqVecTime << " ms\n";
    cout << "Parallel Time (CUDA): " << parVecTime << " ms\n";

    // ========== Matrix Multiplication ==========
    int M = 256, N = 256, K = 256;
    int *matA = new int[M * N];
    int *matB = new int[N * K];
    int *matC_seq = new int[M * K];
    int *matC_par = new int[M * K];

    for (int i = 0; i < M * N; ++i)
        matA[i] = i % 100;
    for (int i = 0; i < N * K; ++i)
        matB[i] = (i * 2) % 100;

    int *d_matA, *d_matB, *d_matC;
    size_t matASize = M * N * sizeof(int);
    size_t matBSize = N * K * sizeof(int);
    size_t matCSize = M * K * sizeof(int);

    cudaMalloc(&d_matA, matASize);
    cudaMalloc(&d_matB, matBSize);
    cudaMalloc(&d_matC, matCSize);

    cudaMemcpy(d_matA, matA, matASize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, matBSize, cudaMemcpyHostToDevice);

    dim3 threads2D(16, 16);
    dim3 blocks2D((K + 15) / 16, (M + 15) / 16);

    // Sequential matrix multiplication
    start = high_resolution_clock::now();
    sequentialMatrixMultiply(matA, matB, matC_seq, M, N, K);
    end = high_resolution_clock::now();
    auto seqMatTime = duration_cast<milliseconds>(end - start).count();

    // Parallel matrix multiplication (GPU)
    start = high_resolution_clock::now();
    matrixMultiply<<<blocks2D, threads2D>>>(d_matA, d_matB, d_matC, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(matC_par, d_matC, matCSize, cudaMemcpyDeviceToHost);
    end = high_resolution_clock::now();
    auto parMatTime = duration_cast<milliseconds>(end - start).count();

    cout << "\n===== Matrix Multiplication Results =====\n";
    cout << "Sequential Time: " << seqMatTime << " ms\n";
    cout << "Parallel Time (CUDA): " << parMatTime << " ms\n";
    cout << "Speedup: " << (float)seqMatTime / parMatTime << "x\n";

    // ========== Cleanup ==========
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_seq;
    delete[] h_C_par;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] matA;
    delete[] matB;
    delete[] matC_seq;
    delete[] matC_par;
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    return 0;
}