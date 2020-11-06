/*
Compiling with hipcc:
hipcc mat_add.cpp -o mat_add -std=c++11
./mat_add
Sample Output:
[Enter size of matrix]
100
[matrix addition of 100 elements]
Copy input data from the host memory to the HIP device
HIP kernel launch with dimension (7, 7) blocks of dimension (16, 16) threads
Time taken for addition : 21 microseconds
Copy output data from the HIP device to the host memory
Done
*/

// Matrix addition using HIP C++
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

__global__ void matrixAdd(float **A, float **B, float **C, int numElements)
{
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if (i < numElements && j< numElements)
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main(void)
{

    // Print the matrix length to be used, and compute its size
    int numElements;
    printf("[Enter size of matrix]\n");
    scanf("%d",&numElements);
    
    size_t size = numElements * numElements * sizeof(float);
    printf("[matrix addition of %d elements]\n", numElements);

    // Allocate the host input matrix A
    float *h_A = (float *)malloc(size);

    // Allocate the host input matrix B
    float *h_B = (float *)malloc(size);

    // Allocate the host output matrix C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrix
    for (int i = 0; i < numElements; ++i)
        for (int j= 0; j < numElements; ++j)
    {
        {
            h_A[i * numElements + j] = rand()/(float)RAND_MAX;
            h_B[i * numElements + j] = rand()/(float)RAND_MAX;
        }
    }   

    // Allocate the device input matrix A
    float **d_A = NULL;
    hipMalloc((void **)&d_A, size);

    // Allocate the device input matrix B
    float **d_B = NULL;
    hipMalloc((void **)&d_B, size);



    // Allocate the device output matrix C
    float **d_C = NULL;
    hipMalloc((void **)&d_C, size);


    // Copy the host input matrix A and B in host memory to the device input matrix in
    // device memory
    printf("Copy input data from the host memory to the HIP device\n");
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);

   

    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Specfic number of threads per block and number of 
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numElements -1) / threadsPerBlock.x+1, (numElements-1)/ threadsPerBlock.y+1);

    printf("HIP kernel launch with dimension (%d, %d) blocks of dimension (%d, %d) threads\n", blocksPerGrid.x,blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    // Launch the matrix Add HIP Kernel

    auto start = high_resolution_clock::now();// Calculate Execution Time
    hipLaunchKernelGGL(matrixAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, numElements);    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken for addition : "<< duration.count() << " microseconds"<<"\n";

    

    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the HIP device to the host memory\n");
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);


    // Free device global memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);


    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // hipDeviceReset causes the driver to clean up all state.
    hipDeviceReset();

    printf("Done\n");
    return 0;
}
