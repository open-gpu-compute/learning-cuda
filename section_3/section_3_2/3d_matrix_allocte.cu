#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
// The following code sample allocates a width x height x 
// depth 3D array of floating-point values and shows how 
// to loop over the array elements in device code
// Device code
__global__ void matrix_allocate(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* deice_pointer = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = deice_pointer + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
// Host code
int main()
{
    int width = 128, height = 128, depth = 128;
    cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                        height, depth);
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    matrix_allocate<<<100, 512>>>(devPitchedPtr, width, height, depth);
}