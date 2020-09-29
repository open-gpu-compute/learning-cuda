# Section 2.3
## Memory Heirheachy

- Each thread has a private local memory. Each block has a private memory which is shared between all the threads in the block. Every thread has access to the global memory of the GPU. 
- The constant and texture memory are read-only memory that can be accessed by all the threads. The global, constant and texture are same across kernel launches by an application.
