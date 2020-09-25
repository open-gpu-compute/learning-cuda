# Section 2.1
### Kernels
------
- The functions executed on the GPU are known as kernels. Kernels can be executed N times in parallel by N different CUDA threads.
- In CUDA terminology,CPU and GPU are known as host and device respectively.
- A kernel is defined using `__global__` declaration specifier. The number of CUDA threads that execute the kernel are defined by `<<<....>>>` operator.
- Each thread that executes a kernel is given a `threadIdx` that can be accessed using built-in variables.
------
### Compiling CUDA C++ code:
-------
- NVCC is a compiler used for compiling CUDA C++ code. NVCC comes as sub-package with CUDA toolkit. More on NVCC in Section 3. 
- CUDA toolkit can be installed using this [link](https://docs.nvidia.com/cuda/#installation-guides)
- Check if NVCC is installed properly using `nvcc --version`. You might have to add CUDA path to your `bashrc`.
```
export PATH=/usr/local/cuda-<cuda_version>/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-<cuda-version>/lib64:$LD_LIBRARY_PATH
```
- Compile the program and make an executable file using ` nvcc <filename.cu> -o <filename> ` 
- Run the executable file using `./<filename>`