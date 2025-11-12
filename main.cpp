#include <iostream>
#include <type_traits>
#include "make_callable_t.hpp"

class Foo {
    int m_x;
public:
    Foo(int x) { m_x = x; }
    Foo(const Foo &) = default;

    void bar() {
        std::cout << "bar is called " << m_x << std::endl;
    }
};

int add(int x, int y) {
    return x + y;
}

#ifdef BUILD_WITH_CUDA
void test_cuda_cap();
#endif

    
int main() {

    static_assert(std::is_base_of<Foo, make_callable_t<&Foo::bar>>::value);
    static_assert(std::is_convertible<Foo, make_callable_t<&Foo::bar>>::value);
    static_assert(std::is_convertible<make_callable_t<&Foo::bar>, Foo>::value);

    Foo a(1);
    make_callable_t<&Foo::bar> b(a);
    b();

    using Add = make_callable_t<&add>;
    std::cout << Add()(1,2) << std::endl;

#ifdef BUILD_WITH_CUDA
    test_cuda_cap();
#endif

    return 0;
}