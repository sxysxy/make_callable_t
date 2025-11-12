#include <iostream>
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
    
int main() {
    using Foobar = make_callable_t<&Foo::bar>;
    Foo a(1);
    Foobar b(a);
    using Add = make_callable_t<&add>;
    std::cout << Add()(1,2) << std::endl;
    return 0;
}