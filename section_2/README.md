# Section 2.1
## Kernels

- The functions executed on the GPU are known as kernels. Kernels can be executed N times in parallel by N different CUDA threads.
- In CUDA terminology,CPU and GPU are known as host and device respectively.
- A kernel is defined using `__global__` declaration specifier. The number of CUDA threads that execute the kernel are defined by `<<<....>>>` operator.
- Each thread that executes a kernel is given a `threadIdx` that can be accessed using built-in variables.

## Compiling CUDA C++ code:

- NVCC is a compiler used for compiling CUDA C++ code. NVCC comes as sub-package with CUDA toolkit. More on NVCC in Section 3. 
- CUDA toolkit can be installed using this [link](https://docs.nvidia.com/cuda/#installation-guides)
- Check if NVCC is installed properly using `nvcc --version`. You might have to add CUDA path to your `bashrc`.
```
export PATH=/usr/local/cuda-<cuda_version>/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-<cuda-version>/lib64:$LD_LIBRARY_PATH
```
- Compile the program and make an executable file using ` nvcc <filename.cu> -o <filename> ` 
- Run the executable file using `./<filename>`
# Section 2.2
## Thread Heirheachy
-------
- CUDA organizes threads into groups called "thread blocks" and further organizes these "thread blocks" into a grid structure.
- Threads and threads blocks can be arranged in  1-dimension, 2-dimension or 3-dimension. This provides a natural way to perform computation on vector, matrix or volume.
- The syntax of kernel execution configuration is as follows
`<<< M , T >>>` ,which indicate that a kernel launches with a grid of M thread blocks. Each thread block has T parallel threads. Both M and T can be int or dim3(built-in structure to pass three dimensions).
- `__syncthreads()`  lets you synchronize thread execution to coordinate memory access. `__syncthreads()` acts as a wall where all threads must wait until every thread reaches that point.
- Thread Heirheachy in CUDA :

 ![grid of thread blocks](./images/grid-of-thread-blocks.png)
 # Section 2.3
## Memory Heirheachy

- Each thread has a private local memory. Each block has a private memory which is shared between all the threads in the block. Every thread has access to the global memory of the GPU. 
- The constant and texture memory are read-only memory that can be accessed by all the threads. The global, constant and texture are same across kernel launches by an application.
# Section 2.4
## Heterogeneous Programming 

- Kernels are executed on the device(GPU), and the rest of the C++program runs on the host (CPU).
- Device and host have separate memory spaces in the DRAM, known as device memory and host memory respectively. The C++ program manages global, constant and texture memory visible to threads.
- Unified Memory in CUDA bridges the gap between host and device memory. Memory can be accessed from all GPUs and CPUs using a common address space.

 ![heterogeneous-programming](./images/heterogeneous-programming.png)

