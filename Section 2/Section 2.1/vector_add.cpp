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