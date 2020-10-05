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
