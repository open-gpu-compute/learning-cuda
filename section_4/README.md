# Hardware Implementations
- NVIDIA GPUs are built using an array of Streaming Multiprocessors(SM).
- When a host executes a kernel grid, blocks are enumerated and assigned to different SMs that execute block threads concurrently.
## SIMT Architecture 
- SIMT (Single-Instruction, Multiple-Thread) is an architecture used by SMs to execute hundreds of threads concurrently.
- The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps.
- When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps, and each warp gets scheduled by a warp scheduler for execution.
- GPUs uses Independent Thread Scheduling that maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity.
## Hardware Multithreading 
- Each multiprocessor has a set of 32-bit registers that are partitioned among warps, and shared memory that is partitioned among the thread blocks.
- The number of blocks and warps that can reside processed together on SMs for a given kernel depends on the number of registers and shared memory used by the kernel and the number of registers and shared memory available on the multiprocessor.

# Performance Guidlines
## Overall Performance
- CUDA optimizes the following features of a GPU: 
    - Maximize parallel execution 
    - Optimize memory usage 
    - Optimize instruction usage 
## Maximize utilization 
- To maximize utilization, the application should be structured in a way that it keeps most of the elements of the GPU busy all the time. 
### Application Level maximize utilization 
- The application should use asynchronous functions calls and streams to maximize parallel execution between the host, the devices, and the bus connecting the host to the devices. 
### Device Level maximize utilization 
- At a lower level, the application should maximize parallel execution between the multiprocessors of a device. 
### Multiprocessor Level maximize utilization 
- At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor. 
- The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the latency.
- All warp schedulers should always have some instruction to issue for some warp at every clock cycle during the latency period to achieve full utilization.
## Maximize Memory Throughput 
- Minimizing data transfers with low bandwidth can help in maximizing the memory throughput of an application.
## Maximize Instruction Throughput 
- Minimize the use of arithmetic instructions with low throughput; this includes trading precision for speed when it does not affect the result, such as using single-precision instead of double-precision.
- Minimize divergent warps caused by control flow instructions 
- Reduce the number of instructions
