#ifndef COMPLEX_HEADER_H
#define COMPLEX_HEADER_H
#include <stan/math.hpp>

template<typename T>
class Complex
{
    private:
    T Real;
    T Imag;

    public:
    Complex(T _Real = 0.0, T _Imag = 0.0);
    Complex(const Complex<T> &z);

    //getter
    T real() const;
    T imag() const;
    //setter
    void real(T _Real);
    void imag(T _Imag);

    //conjugate
    Complex<T> conj() const;

    //norm
    T squaredNorm() const;

    //unary operators
    Complex<T> &operator=(const Complex<T> &z);
    Complex<T> &operator=(const T &x);
    Complex<T> &operator=(const std::complex<double> &z);
    Complex<T> &operator+=(const Complex<T> &z);
    //Complex<T> &operator+=(T x);
    Complex<T> &operator-=(const Complex<T> &z);
    Complex<T> &operator-=(const T &x);
    Complex<T> &operator*=(const Complex<T> &z);
    //Complex<T> &operator*=(T x);
    Complex<T> &operator/=(const Complex<T> &z);
    Complex<T> &operator/=(const T &x);
};
//unary operators
template<typename T>
Complex<T> operator-(const Complex<T> &z)
{
  Complex<T> zcopy(-z.real(), -z.imag());
  return zcopy;
}
//binary operators
template<typename T>
std::ostream &operator<<(std::ostream &os, const Complex<T> &z);
template<typename T>
bool operator==(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
bool operator!=(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator+(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator-(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator-(const Complex<T> &LHS, const T &RHS);
template<typename T>
Complex<T> operator-(const T &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator*(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator/(const Complex<T> &LHS, const Complex<T> &RHS);
template<typename T>
Complex<T> operator/(const Complex<T> &LHS, const T &RHS);
template<typename T>
Complex<T> operator/(const T &LHS, const Complex<T> &RHS);

//Eigen compatability
template<typename T>
inline Complex<T> conj(const Complex<T> &z)  { return z.conj(); }
template<typename T>
inline T real(const Complex<T> &z)  { return z.real(); }
template<typename T>
inline T imag(const Complex<T> &z)    { return z.imag(); }
template<typename T>
inline T abs(const Complex<T> &z)  { return stan::math::sqrt(z.squaredNorm()); }
template<typename T>
inline T abs2(const Complex<T> &z)  { return z.squaredNorm(); }

namespace Eigen
{
    template<> struct NumTraits<Complex<double>> : NumTraits<std::complex<double>>
    {
        typedef double Real;
        typedef Complex<double> NonInteger;
        typedef double Literal;
        typedef Complex<double> Nested;

        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 2 * NumTraits<double>::ReadCost,
            AddCost = 2 * NumTraits<double>::AddCost,
            MulCost = 4 * NumTraits<double>::MulCost + 2 * NumTraits<double>::AddCost
        };
    };

    template<> struct NumTraits<Complex<stan::math::var>> : NumTraits<std::complex<double>>
    {
        typedef stan::math::var Real;
        typedef Complex<stan::math::var> NonInteger;
        typedef stan::math::var Literal;
        typedef Complex<stan::math::var> Nested;

        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 2 * NumTraits<double>::ReadCost,
            AddCost = 2 * NumTraits<double>::AddCost,
            MulCost = 4 * NumTraits<double>::MulCost + 2 * NumTraits<double>::AddCost
        };
    };
}
#endif
