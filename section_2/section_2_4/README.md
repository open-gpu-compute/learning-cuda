# Section 2.4
## Heterogeneous Programming 

- Kernels are executed on the device(GPU), and the rest of the C++program runs on the host (CPU).
- Device and host have separate memory spaces in the DRAM, known as device memory and host memory respectively. The C++ program manages global, constant and texture memory visible to threads.
- Unified Memory in CUDA bridges the gap between host and device memory. Memory can be accessed from all GPUs and CPUs using a common address space.

 ![heterogeneous-programming](../images/heterogeneous-programming.png)
