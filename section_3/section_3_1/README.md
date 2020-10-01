# NVCC Compilation

- Kernels can be either written using a higher-level language like C++ or using CUDA instruction set architecture, called PTX.
- In both cases, `nvcc` is used to convert Kernels into binary code.
## Offline Compilation

- `nvcc` separates the host code from the device code.
- The separated device code is compiled into PTX and/or binary code.
- `nvcc` also removes CUDA built-in syntax and variables like `<<<...>>>` from the host code.
- The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting `nvcc` invoke the host compiler during the last compilation stage.
## Just-in-Time Compilation

- PTX code loaded by an application at runtime can be compiled further to binary code by the device driver. This is called just-in-time compilation. Just-in-time compilation increases application load time but allows the application to benefit from any new compiler improvements coming with each new device driver.
- NVRTC compiler can be used to compile CUDA C++ device code to PTX at runtime.
##  Binary Compatibility

- Compute capability is a version number, also called "SM version", that tells the features supported by a GPU. It is used by applications at runtime to determine which features are available on the device.
- Binary code is architecture-specific and different for different compute capabilities.
- Compute capability can be specified in NVCC while compiling the code using compiler option `code`. For example, compiling with `-code=sm_35` produces binary code for devices of compute capability 3.5.
## PTX Compatibility

- PTX instructions are also architecture-specific. Some PTX instructions are only supported by higher versions of compute capability.
- The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to PTX code.
## Application Compatibility

- For an application to be compatibility with a GPU, it must load binary or PTX code that is compatible with this compute capability as described in the above sections.For example,
```
nvcc vector_add.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
```
generates binary code compatible with compute capability 5.0 and 6.0.
## C++ and 64-bit Compatibility

- Host code has full C++ support, while only a subset of C++ is supported for device code.
- The 64-bit version of nvcc can compile device code in 32-bit mode using  `-m32` compiler option.
