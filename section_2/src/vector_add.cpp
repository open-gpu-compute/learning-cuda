/* 
Compiling with g++:
g++ vector_add.cpp -o vector_add 
./vector_add
Sample Output :
[Enter size of vector]
1000
[Vector addition of 1000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 2 blocks of 512 threads
Time taken by function : 40 microseconds
Copy output data from the CUDA device to the host memory
Done
*/

// Vector addition using only CPU
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i=0;
    while (i < numElements)
    {
        C[i] = A[i] + B[i];
        i++;
    }
}

int main(void)
{

    // Print the vector length to be used, and compute its size
    int numElements;
    scanf("%d",&numElements);
    
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements using only CPU]\n", numElements);

    // Allocate the host input vector A
    float *A = (float *)malloc(size);

    // Allocate the host input vector B
    float *B = (float *)malloc(size);

    // Allocate the host output vector C
    float *C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }
    auto start = high_resolution_clock::now();
    vectorAdd(A,B,C,numElements);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function : "<< duration.count() << " microseconds"<<"\n";
    
    // Free CPU memory
    free(A);
    free(B);
    free(C);


    printf("Done\n");
    return 0;
}