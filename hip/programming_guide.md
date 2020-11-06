## Programming Guide
```

__global__ void helloworld(char* in, char* out)
{
	int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
	out[num] = in[num] + 1;
}

int main(int argc, char* argv[])
{

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

	/* Initial input,output for the host and create memory objects for the kernel*/
	const char* input = "GdkknVnqkc";
	size_t strlength = strlen(input);
	cout << "input string:" << endl;
	cout << input << endl;
	char *output = (char*) malloc(strlength + 1);

	char* inputBuffer;
	char* outputBuffer;
	hipMalloc((void**)&inputBuffer, (strlength + 1) * sizeof(char));
    hipMalloc((void**)&outputBuffer, (strlength + 1) * sizeof(char));

    hipMemcpy(inputBuffer, input, (strlength + 1) * sizeof(char), hipMemcpyHostToDevice);

	hipLaunchKernelGGL(helloworld,
                  dim3(1),
                  dim3(strlength),
                  0, 0,
                  inputBuffer ,outputBuffer );

	hipMemcpy(output, outputBuffer,(strlength + 1) * sizeof(char), hipMemcpyDeviceToHost);

    hipFree(inputBuffer);
    hipFree(outputBuffer);

	output[strlength] = '\0';	//Add the terminal character to the end of output.
	cout << "\noutput string:" << endl;
	cout << output << endl;

	free(output);

	std::cout<<"Passed!\n";
	return SUCCESS;
}
```
Following tutorial goes through rotates a string by 1 ASCII character. You can find the full code at `src/string.cpp`


```
#include <hip/hip_runtime.h>
```
Include headers for HIP runtime libraries


```
__global__ void helloworld(char* in, char* out)
{
		int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
		out[num] = in[num] + 1;
}
```

Supported `__global__` functions are executed on the device and called ("launched") from the host.
HIP `__global__` functions must have a void return type.
Apart from `__global__` functions, there are two more types of functions in HIP:
* `__device__` :Supported `__device__` functions are executed on the device, called from the device only.

* `__host__`: Supported `__host__` functions are executed on the host and called from the host
`__host__` can combine with `__device__`, in which case the function compiles for both the host and device. These functions cannot use the HIP grid coordinate function.
	 
```
hipDeviceProp_t devProp;
```
`hipDeviceProp_t` is a struct use to store device properties similar to CUdevprop
```
hipGetDeviceProperties(&devProp, 0);
```
`hipGetDeviceProperties` returns properties for a selected device.
```
hipMalloc((void**)&inputBuffer, (strlength + 1) * sizeof(char));
hipMalloc((void**)&outputBuffer, (strlength + 1) * sizeof(char));
hipMemcpy(inputBuffer, input, (strlength + 1) * sizeof(char), hipMemcpyHostToDevice);
```
The runtime provides built-in functions to allocate, deallocate and copy device memory. It also provides functions to transfer data between the device and host memory. The device memory can be allocated as linear memory . Linear memory uses a single unified address space, which allows separately allocated entities to address each other via pointers. Linear memory is allocated using `hipMalloc()` and freed using `hipFree()`, and data transfer between host memory and device memory is done using `hipMemcpy()`.
```	 
hipLaunchKernelGGL(helloworld,dim3(1),dim3(strlength),0, 0,inputBuffer ,outputBuffer );
```
	 
`__global__` functions are often referred to as kernels, and calling one is termed launching the kernel. These functions require the caller to specify an "execution configuration" that includes the grid and block dimensions. The execution configuration can also include other information for the launch, such as the amount of additional shared memory to allocate and the stream where the kernel should execute. HIP introduces a standard C++ calling convention to pass the execution configuration to the kernel in addition to the Cuda `<<< >>>` syntax. In HIP,
Kernels launch with either `<<< >>>` syntax or the "hipLaunchKernel" function
The first five parameters to hipLaunchKernel are the following:
* symbol kernelName: the name of the kernel to launch. To support template kernels which contains "," use the HIP_KERNEL_NAME macro. The hipify tools insert this automatically.
* `dim3 gridDim`: 3D-grid dimensions specifying the number of blocks to launch.
* `dim3 blockDim`: 3D-block dimensions specifying the number of threads in each block.
* `size_t dynamicShared`: amount of additional shared memory to allocate when launching the kernel (see shared)
* `hipStream_t`: stream where the kernel should execute. 
`dim3` is a three-dimensional integer vector type commonly used to specify grid and group dimensions. Unspecified dimensions are initialized to 1.
### Mapped Memory
A block of page-locked host memory can also be mapped into the address space of the device by passing flag`hipHostAllocMapped` to `hipHostAlloc()` or by passing flag `hipHostRegisterMapped` to `hipHostRegister()`. Such a block has therefore in general two addresses: one in host memory that is returned by `hipHostMalloc()` or `malloc()`, and one in device memory that can be retrieved using `hipHostGetDevicePointer()` and then used to access the block from within a kernel.

### Matrix Multiplication Using Shared Memory
Below contains the code for matrix multiplication in HIP using shared memory. Full code can be found at `src/mat_mul.src`

```
 __global__ void  MatMulKernelSharedMemory(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = hipBlockIdx_y;
    int blockCol = hipBlockIdx_x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = hipThreadIdx_y;
    int col = hipThreadIdx_x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
```
`__shared__` : Shared memory is allocated using `__shared__` memory space specifier. Its is faster than global memory and reduce global memory access calls.

### Stream
Streams are a sequence of commands that execute in order. There can be multiple streams executed on different kernels. If kernel launches do not specify a stream, the commands are run on default stream, known as stream 0. The following code sample creates two streams.Each of these streams is defined by the following code sample as a sequence of one memory copy from host to device and one memory copy from device to host:
```
hipStream_t stream[2];
for (int i = 0; i < 2; ++i)
    hipStreamCreate(&stream[i]);
float* hostPtr;
hipMallocHost(&hostPtr, 2 * size);
for (int i = 0; i < 2; ++i) {
    hipMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
for (int i = 0; i < 2; ++i)
    hipStreamDestroy(stream[i]);
```
`cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed. cudaStreamSynchronize()takes a stream as a parameter and waits until all preceding commands in the given stream have completed.

## Events 
The HIP runtime provides a way to monitor the device's progress by letting the application asynchronously record events at any point in the program, and query when these events are completed.
``` 
hipEvent_t start, stop; 
hipEventCreate(&start); 
hipEventCreate(&stop); 
```
## Multi-Device System 
Similar to CUDA, HIP support multiple devices for a host. A certain device can be selected for a certain stream.
A host thread can set the device it operates on at any time by calling `hipSetDevice()`. 
```
int deviceCount;
hipGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    hipDeviceProp deviceProp;
    hipGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}

```
This code lets you print properties of device on the system.

### Peer-to-Peer Memory Access
In a system with multiple devices, devices can address each other's memory depending upon their compute capability.
This peer-to-peer memory access feature is supported between two devices if `hipDeviceCanAccessPeer()` returns true for these two devices. 
A unified address space is used for both devices, so the same pointer can be used to address memory from both devices as shown in the code sample below
```
hipSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
hipMalloc(&p0, size);              // Allocate memory on device 0
hipSetDevice(1);                   // Set device 1 as current
hipDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                    // with device 0
```
