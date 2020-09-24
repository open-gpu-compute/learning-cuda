# Section 2.2
### Thread Heirheachy
-------
- CUDA organizes threads into groups called "thread blocks" and further organizes these "thread blocks" into a grid structure.
- Threads and threads blocks can be arranged in  1-dimension, 2-dimension or 3-dimension. This provides a natural way to perform computation on vector, matrix or volume.
- The syntax of kernel execution configuration is as follows
`<<< M , T >>>` ,which indicate that a kernel launches with a grid of M thread blocks. Each thread block has T parallel threads. Both M and T can be int or dim3(built-in structure to pass three dimensions).
- `__syncthreads()`  lets you synchronize thread execution to coordinate memory access. `__syncthreads()` acts as a wall where all threads must wait until every thread reaches that point.
- Thread Heirheachy in CUDA :

 ![alt text](grid-of-thread-blocks.png)
