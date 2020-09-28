# Introduction

## Comparasion between GPU and CPU

- The Graphics Processing Unit (GPU) provides higher memory bandwidth and instructions throughput as compared to the CPU within the same power usage.
- GPUs have more transistors dedicated to data processing (e.g. float point computation), while CPUs have more transistors decided to cache and control flow.
- GPUs are specifically designed for parallel computing. Recent GPUs can execute thousands of threads (a sequence of operations) parallelly, but they have lower single-thread performance than CPUs.
- Archituture of CPU and GPU is shown below

![cpu vs gpu architecture](images/GPU_CPU_architecture.PNG)

## What is CUDA

- CUDA (Compute Unified Device Architecture) is a platform and programming model for NVIDIA GPUs.
- CUDA exposes GPU to general-purpose computing, thus parallelism of GPUs can be exploited. CUDA provides C++ as a higher-level language for programming and managing GPUs.
- CUDA is designed to support various languages and application programming interfaces.

## CUDA as a scalable programming model

- CUDA transparently scales GPU's parallelism using only three core abstractions, namely, a hierarchy of thread groups, shared memories, and barrier synchronization.Thus making the learning curve for the userr easier.
- These abstractions provide both task parallelization (running different tasks on the same data) and data parallelization (running a single task on different components of data).
- CUDA can schedule each block of threads on any multiprocessor within the GPU in any order, concurrently or sequentially. Once compiled, a CUDA program can run on any number of multiprocessors.
- Automatic Scalability of CUDA in different GPUs:

![alt text](images/automatic-scalability.png)
