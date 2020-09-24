# Section 2.1
### Kernels
------
- The functions executed on the GPU are known as kernels. Kernels are executed N times in parallel by N different CUDA threads.
- In CUDA terminology,CPU and GPU are known as host and device respectively.
- A kernel is defined using `__global__` declaration specifier. The number of CUDA threads that execute the kernel are defined by `<<<....>>>`.
- Each thread that executes a kernel has a unique `threadIdx` that can be accessed using built-in variables
