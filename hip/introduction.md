# Introduction
## What is ROCm / HCC?
ROCm / HCC is AMD's Single-source C++ framework for GPGPU programming. In effect: HCC is a CLang based compiler, which compiles your code in two passes. It compiles an x86 version of your code AND a GPU version of your code.
Because the same compiler processes both x86 and GPU code, it ensures that all data-structures are compatible. With AMD's HSA project of the past, even pointers remain consistent between the codesets, allowing the programmer to easily transition between CPU and GPU code.
In effect, ROCm / HCC is AMD's full attempt at a CUDA-like C++ environment. While OpenCL requires you to repeat yourself with any shared data-structure (in C nonetheless), HCC allows you to share pointers, classes, and structures between the CPU and GPU code.

## What is Heterogeneous-Computing Interface for Portability (HIP)?
 It's a C++ dialect designed to ease conversion of Cuda applications to portable C++ code. It provides a C-style API and a C++ kernel language. The C++ interface can use templates and classes across the host/kernel boundary.
The HIPify tool automates much of the conversion work by performing a source-to-source transformation from Cuda to HIP. HIP code can run on AMD hardware (through the HCC compiler) or Nvidia hardware (through the NVCC compiler) with no performance loss compared with the original Cuda code.
Programmers familiar with other GPGPU languages will find HIP very easy to learn and use. AMD platforms implement this language using the HC dialect described above, providing similar low-level control over the machine.
Similar to CUDA, HIP transparently scales GPU's parallelism using only three core abstractions, making the learning curve easier:

* a hierarchy of thread groups
* shared memories
* barrier synchronization.
## Installing HIP/HCC
HIP can be installed from the following link:
https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md
## Quick Hands-On 
Below contains the code for vector addition using a CUDA kernel. 
```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
```
Compile the code and run it as:
```
$ nvcc src/vector_add.cu -o vector_add_gpu -std=c++11
$ ./vector_add_gpu
Enter number of elements in your vector:
1000000
[Vector addition of 1000000 elements]
[Copy input data from the host memory to the CUDA device]
[CUDA kernel launch with 1954 blocks of 512 threads]
Time taken by function : 62 microseconds
[Copy output data from the CUDA device to the host memory]
Done
```
The same code in HIP will be written as :
```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
```
Compile the code and run it as:
```
$ hipcc src/vector_add.cpp -o vector_add_gpu -std=c++11
$ ./vector_add_gpu
Enter number of elements in your vector:
1000000
[Vector addition of 1000000 elements]
[Copy input data from the host memory to the HIP device]
[HIP kernel launch with 1954 blocks of 512 threads]
Time taken by function : 86 microseconds
[Copy output data from the HIP device to the host memory]
Done
```

If executed on GPUs with equal computation power, both CUDA and HIP take approximatively equal time to run the function. 
`__global__` functions are executed on GPU and often known as Kernels.
Each thread that executes a kernel is given a `hipthreadIdx_x` that can be accessed using built-in variables.



