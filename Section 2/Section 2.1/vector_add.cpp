// Kernel definition 
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
// host code
int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C); //each of the N threads that execute VecAdd() performs one pair-wise addition
    ...
}