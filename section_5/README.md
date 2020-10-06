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
