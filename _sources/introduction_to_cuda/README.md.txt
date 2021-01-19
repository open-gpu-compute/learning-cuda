# Introduction

## Comparasion between GPU and CPU

The Graphics Processing Unit (GPU) provides higher memory bandwidth and instructions throughput as compared to the CPU within the same power usage. They have more transistors dedicated to data processing (e.g. float point computation), while CPUs have more transistors decided to cache and control flow.

GPUs are specifically designed for parallel computing. Recent GPUs can execute thousands of threads (a sequence of operations) parallelly, but they have lower single-thread performance than CPUs.

Archituture of CPU and GPU is shown below:

![cpu vs gpu architecture](images/GPU_CPU_architecture.PNG)

## CUDA, a scalable programming model

CUDA (Compute Unified Device Architecture) is a platform and programming model for NVIDIA GPUs. It exposes GPU to general-purpose computing, thus parallelism of GPUs can be exploited. CUDA provides C++ as a higher-level language for programming and managing GPUs. It is designed to support various languages and application programming interfaces.

CUDA transparently scales GPU's parallelism using only three core abstractions, making the learning curve easier:

* a hierarchy of thread groups
* shared memories
* barrier synchronization.

These abstractions provide both task parallelization (running different tasks on the same data) and data parallelization (running a single task on different components of data).

CUDA can schedule each block of threads on any multiprocessor within the GPU in any order, concurrently or sequentially. Once compiled, a CUDA program can run on any number of multiprocessors.

Automatic Scalability of CUDA in different GPUs:

![Automatic Scalability](images/automatic-scalability.png)
