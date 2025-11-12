#include "cuda_runtime.h"
#include "make_callable_t.hpp"
#include <stdio.h>
#include <iostream>

template<class Fn>
__global__ void test_kernel(Fn f) {
    // Call operator()
    f(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y))); 
}

struct TestCUDA {
    int64_t base = 0;
    __device__ void test(int64_t i) {  // just output the thread index + base
        printf("%lld ", base + i);
    }
};

void test_cuda_cap() {
    cudaSetDevice(0);
    
    TestCUDA a{-256};
    test_kernel<<<16, 16>>>(make_callable_t<&TestCUDA::test>(a) ); // Wrap the object a

    if(cudaDeviceSynchronize() != cudaSuccess) {  // should not error
        std::cout << "CUDA Error:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
}