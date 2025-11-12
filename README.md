# make_callable_t

A header-only C++ template helper to wrap callable(pseudo function) class, with zero-cost abstraction, CUDA compatibility.

## Introduce the Library

**Header-only**, just include make_callable_t.hpp, requires c++20(or above)

By default, it will introduce a make_callable_t template into the global namespace. 

## Features

- **ZERO-COST ABSTRACTION**: Do not introduce additional runtime cost.

- Wrap a member function to a callable pseudo class, ensuring convertible between the origin class and the wrapper class, e.g.:
```cpp
    static_assert(std::is_base_of<Foo, make_callable_t<&Foo::bar>>::value);
    static_assert(std::is_convertible<Foo, make_callable_t<&Foo::bar>>::value);
    static_assert(std::is_convertible<make_callable_t<&Foo::bar>, Foo>::value);
```
- Wrap a free functions to a callable pseudo class.
```cpp
int add(int x, int y) {
    return x + y;
}
int main() {
    using Add = make_callable_t<&add>;
    std::cout << Add()(1,2) << std::endl; // OK, 3
    return 0;
}
```
- Compatible with CUDA device function(requires nvcc --expt-relaxed-constexpr). The wrapper does not explicitly declare \_\_host\_\_ or \_\_device\_\_, it relies on relaxed policy of constexpr.
```cpp
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
```

## Eamples

see [main.cpp](main.cpp)

see [test_cuda_cap](test_cuda_cap.cu)

## Build the Example

Without CUDA:

```bash
bash build.sh
```

With CUDA:

```bash
build_with_cud.sh
```

## LICENSE
[DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE (version 2)](LICENSE)

The single source code [make_callable_t.hpp](make_callable_t.hpp) includes this license, you can use and distribute that single file.