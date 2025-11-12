#pragma once
#include <type_traits>
#include <utility>

namespace make_callable_details {

template <typename Base, auto Method>
struct mf_callable_t : Base {
    static_assert(std::is_member_function_pointer_v<decltype(Method)>);
    using Base::Base; 

    constexpr mf_callable_t(const Base& other)
        noexcept(std::is_nothrow_copy_constructible_v<Base>)
        requires std::is_copy_constructible_v<Base>
        : Base(other) {}

    constexpr mf_callable_t(Base&& other)
        noexcept(std::is_nothrow_move_constructible_v<Base>)
        requires std::is_move_constructible_v<Base>
        : Base(std::move(other)) {}

    constexpr mf_callable_t& operator=(const Base& other)
        noexcept(noexcept(static_cast<Base&>(*this) = other))
        requires std::is_copy_assignable_v<Base>
    { static_cast<Base&>(*this) = other; return *this; }

    constexpr mf_callable_t& operator=(Base&& other)
        noexcept(noexcept(static_cast<Base&>(*this) = std::move(other)))
        requires std::is_move_assignable_v<Base>
    { static_cast<Base&>(*this) = std::move(other); return *this; }

    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) &
        noexcept(noexcept((std::declval<Base&>().*Method)(std::forward<Args>(args)...))) {
        return (this->*Method)(std::forward<Args>(args)...);
    }
    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) const&
        noexcept(noexcept((std::declval<const Base&>().*Method)(std::forward<Args>(args)...))) {
        return (this->*Method)(std::forward<Args>(args)...);
    }
    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) &&
        noexcept(noexcept((std::declval<Base&&>().*Method)(std::forward<Args>(args)...))) {
        return (std::move(*this).*Method)(std::forward<Args>(args)...);
    }
    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) const&&
        noexcept(noexcept((std::declval<const Base&&>().*Method)(std::forward<Args>(args)...))) {
        return (std::move(*this).*Method)(std::forward<Args>(args)...);
    }
};

template <auto Function>
struct fn_callable_t {
    static_assert(std::is_pointer_v<decltype(Function)>, "Function must be a function pointer");
    using Fn = std::remove_pointer_t<decltype(Function)>;
    static_assert(std::is_function_v<Fn>, "Function must point to a function");

    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) const
        noexcept(noexcept(Function(std::forward<Args>(args)...))) {
        return Function(std::forward<Args>(args)...);
    }
};

#if defined(__cpp_lib_member_pointer_traits) && __cpp_lib_member_pointer_traits >= 202106L
  #include <functional>
  template <typename MP>
  using member_class_t = typename std::member_pointer_traits<MP>::class_type;
#else
  template <typename> struct member_class;
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...)> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) const> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) &> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) const&> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) &&> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) const&&> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) noexcept> { using type = C; };
  template <typename C, typename R, typename... A>
  struct member_class<R (C::*)(A...) const noexcept> { using type = C; };

  template <typename MP>
  using member_class_t = typename member_class<MP>::type;
#endif

template <auto Target, bool IsMem = std::is_member_function_pointer_v<decltype(Target)>>
struct make_callable_impl;

template <auto Target>
struct make_callable_impl<Target, false> : fn_callable_t<Target> {};

template <auto Target>
struct make_callable_impl<Target, true>
    : mf_callable_t<member_class_t<decltype(Target)>, Target> {
    
    using base_t = member_class_t<decltype(Target)>;
    using self_t = make_callable_impl;
    using mf_callable_t<base_t, Target>::mf_callable_t;

    constexpr make_callable_impl(const base_t& b)
        noexcept(std::is_nothrow_copy_constructible_v<base_t>)
        requires std::is_copy_constructible_v<base_t>
        : mf_callable_t<base_t, Target>(b) {}

    constexpr make_callable_impl(base_t&& b)
        noexcept(std::is_nothrow_move_constructible_v<base_t>)
        requires std::is_move_constructible_v<base_t>
        : mf_callable_t<base_t, Target>(std::move(b)) {}

    constexpr make_callable_impl(const self_t&) = default;
    constexpr make_callable_impl(self_t&&)      = default;

    constexpr self_t& operator=(const base_t& b)
        noexcept(noexcept(static_cast<base_t&>(*this) = b))
        requires std::is_copy_assignable_v<base_t>
    { static_cast<base_t&>(*this) = b; return *this; }

    constexpr self_t& operator=(base_t&& b)
        noexcept(noexcept(static_cast<base_t&>(*this) = std::move(b)))
        requires std::is_move_assignable_v<base_t>
    { static_cast<base_t&>(*this) = std::move(b); return *this; }

};

} // end of namespace make_callable_details 

template <auto Target>
using make_callable_t = make_callable_details::make_callable_impl<Target>;
