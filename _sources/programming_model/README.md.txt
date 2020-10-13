# Programming Model

## Kernels

In CUDA terminology,CPU and GPU are known as host and device respectively. The functions executed on the GPU are known as kernels. Kernels can be executed N times in parallel by N different CUDA threads.A kernel is defined using `__global__` declaration specifier. The number of CUDA threads that execute the kernel are defined by `<<<....>>>` operator.

Each thread that executes a kernel is given a `threadIdx` that can be accessed using built-in variables.

## Thread Heirearchy

CUDA organizes threads into groups called "thread blocks" and further organizes these "thread blocks" into a grid structure. Threads and threads blocks can be arranged in  1-dimension, 2-dimension or 3-dimension. This provides a natural way to perform computation on vector, matrix or volume.

The syntax of kernel execution configuration is
`<<< M , T >>>`, which indicates that a kernel launches with a grid of M thread blocks. Each thread block has T parallel threads. Both M and T can be int or dim3(built-in structure to pass three dimensions).

`__syncthreads()`  lets you synchronize thread execution to coordinate memory access. `__syncthreads()` acts as a wall where all threads must wait until every thread reaches that point.

 ![grid of thread blocks](./images/grid-of-thread-blocks.png)

## Memory Heirearchy

Each thread has a private local memory. Each block has a private memory which is shared between all the threads in the block. Every thread has access to the global memory of the GPU.

There also exists constant and texture memory which are read-only memory that can be accessed by all the threads. The global, constant and texture are same across kernel launches by an application.

## Heterogeneous Programming

Kernels are executed on the device (GPU), and the rest of the C++ program runs on the host (CPU). Device and host have separate memory spaces in the DRAM, known as device memory and host memory respectively. The C++ program manages global, constant and texture memory visible to threads.

Unified Memory in CUDA bridges the gap between host and device memory. Memory can be accessed from all GPUs and CPUs using a common address space.

 ![heterogeneous-programming](./images/heterogeneous-programming.png)


## Compiling CUDA C++ code:

NVCC is a compiler used for compiling CUDA C++ code. NVCC comes as sub-package with CUDA toolkit. More on NVCC in Section 3.

CUDA toolkit can be installed using this [link](https://docs.nvidia.com/cuda/#installation-guides). Check if NVCC is installed properly using `nvcc --version`. You might have to add CUDA path to your `bashrc`.

```bash
export PATH=/usr/local/cuda-<cuda_version>/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-<cuda-version>/lib64:$LD_LIBRARY_PATH
```

Compile the program and make an executable file using

```
nvcc <filename.cu> -o <filename>
```

Run the executable file using `./<filename>`

## Quick Hands On

### CPU

Let's write a simple program to add two vectors. In standard CPU based code, you would write this as

```cpp
void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i=0;
    while (i < numElements)
    {
        C[i] = A[i] + B[i];
        i++;
    }
}
```

Complete code is in `src/vector_add.cpp`. Compile and run it as

```bash
$ g++ src/vector_add.cpp -o vector_add_cpu
$ ./vector_add_cpu
Enter number of elements in your vector:
1000000
[Vector addition of 1000000 elements using only CPU]
Time taken by function: 9898 microseconds
Done
```

### GPU

On CUDA, you would write this as a kernel:

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

`__global__` decorator indicates that this is a kernel.

Here threads are enumerated using `i` and are executed concurrently. 

Complete code is in `src/vector_add.cu`. Read it for memory allocation and device copy semantics. Compile and run it as

```bash
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

See how massive parallelization helped speed up the computation by a factor of 150.
