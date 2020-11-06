/* 
Compiling with hipcc:
hipcc vector_add.cpp -o vector_add -std=c++11
./vector_add
Sample Output :
[Enter size of vector]
1000
[Vector addition of 1000 elements]
Copy input data from the host memory to the hip device
hip kernel launch with 2 blocks of 512 threads
Time taken by function : 21 microseconds
Copy output data from the hip device to the host memory
Done
*/

// Vector addition on GPU and CPU, using hip C++
#include <stdio.h>
#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{

    // Print the vector length to be used, and compute its size
    int numElements;
    printf("Enter number of elements in your vector: \n");
    scanf("%d",&numElements);
    
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    hipMalloc((void **)&d_A, size);

    // Allocate the device input vector B
    float *d_B = NULL;
    hipMalloc((void **)&d_B, size);

    // Allocate the device output vector C
    float *d_C = NULL;
    hipMalloc((void **)&d_C, size);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("[Copy input data from the host memory to the hip device]\n");
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Launch the Vector Add hip Kernel
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements - 1) / threadsPerBlock + 1;

    printf("[hip kernel launch with %d blocks of %d threads]\n", blocksPerGrid, threadsPerBlock);

    auto start = high_resolution_clock::now();// Calculate Exection Time
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function : "<< duration.count() << " microseconds"<<"\n";

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("[Copy output data from the hip device to the host memory]\n");
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
