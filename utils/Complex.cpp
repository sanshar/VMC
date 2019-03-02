#include "Complex.h"
#include <stan/math.hpp>

//constructor
template<typename T>
Complex<T>::Complex(T _Real, T _Imag) : Real(_Real), Imag(_Imag) {}
template
Complex<double>::Complex(double _real, double _Imag);
template
Complex<stan::math::var>::Complex(stan::math::var _real, stan::math::var _Imag);

//copy constructor
template<typename T>
Complex<T>::Complex(const Complex<T> &z)
{
    Real = z.Real;
    Imag = z.Imag;
}
template
Complex<double>::Complex(const Complex<double> &z);
template
Complex<stan::math::var>::Complex(const Complex<stan::math::var> &z);

//getter
template<typename T>
T Complex<T>::real() const
{
    return Real;
}
template
double Complex<double>::real() const;
template
stan::math::var Complex<stan::math::var>::real() const;

template<typename T>
T Complex<T>::imag() const
{
    return Imag;
}
template
double Complex<double>::imag() const;
template
stan::math::var Complex<stan::math::var>::imag() const;

//setter
template<typename T>
void Complex<T>::real(T _Real)
{
    Real = _Real;
}
template
void Complex<double>::real(double _Real);
template
void Complex<stan::math::var>::real(stan::math::var _Real);

template<typename T>
void Complex<T>::imag(T _Imag)
{
    Imag = _Imag;
}
template
void Complex<double>::imag(double _Imag);
template
void Complex<stan::math::var>::imag(stan::math::var _Imag);

//conjugate
template<typename T>
Complex<T> Complex<T>::conj() const
{
    Complex<T> zcopy(Real, -Imag);
    return zcopy;
}
template
Complex<double> Complex<double>::conj() const;
template
Complex<stan::math::var> Complex<stan::math::var>::conj() const;

//norm
template<typename T>
T Complex<T>::squaredNorm() const
{
    return Real * Real + Imag * Imag;
}
template
double Complex<double>::squaredNorm() const;
template
stan::math::var Complex<stan::math::var>::squaredNorm() const;

//unary operators
template<typename T>
Complex<T> &Complex<T>::operator=(Complex<T> z)
{
    Real = z.Real;
    Imag = z.Imag;
    return *this;
}
template
Complex<double> &Complex<double>::operator=(Complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator=(Complex<stan::math::var> z);

template<typename T>
Complex<T> &Complex<T>::operator=(T x)
{
    Real = x;
    Imag = 0.0;
    return *this;
}
template
Complex<double> &Complex<double>::operator=(double x);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator=(stan::math::var x);

template<typename T>
Complex<T> &Complex<T>::operator=(std::complex<double> z)
{
    Real = z.real();
    Imag = z.imag();
    return *this;
}
template
Complex<double> &Complex<double>::operator=(std::complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator=(std::complex<double> z);

template<typename T>
Complex<T> &Complex<T>::operator+=(Complex<T> z)
{
    Real += z.Real;
    Imag += z.Imag;
    return *this;
}
template
Complex<double> &Complex<double>::operator+=(Complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator+=(Complex<stan::math::var> z);

template<typename T>
Complex<T> &Complex<T>::operator-=(Complex<T> z)
{
    Real -= z.Real;
    Imag -= z.Imag;
    return *this;
}
template
Complex<double> &Complex<double>::operator-=(Complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator-=(Complex<stan::math::var> z);

template<typename T>
Complex<T> &Complex<T>::operator-=(T x)
{
    Real -= x;
    return *this;
}
template
Complex<double> &Complex<double>::operator-=(double x);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator-=(stan::math::var x);

template<typename T>
Complex<T> &Complex<T>::operator*=(Complex<T> z)
{
    T a = Real, b = Imag, c = z.Real, d = z.Imag;
    Real = a * c - b * d;
    Imag = b * c + a * d;
    return *this;
}
template
Complex<double> &Complex<double>::operator*=(Complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator*=(Complex<stan::math::var> z);

template<typename T>
Complex<T> &Complex<T>::operator/=(Complex<T> z)
{
    assert(z.Real || z.Imag);
    T a = Real, b = Imag, c = z.Real, d = z.Imag;
    Real = (a * c + b * d) / z.squaredNorm();
    Imag = (b * c - a * d) / z.squaredNorm();
    return *this;
}
template
Complex<double> &Complex<double>::operator/=(Complex<double> z);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator/=(Complex<stan::math::var> z);

template<typename T>
Complex<T> &Complex<T>::operator/=(T x)
{
    Complex<T> z(x);
    return *this /= z;
}
template
Complex<double> &Complex<double>::operator/=(double x);
template
Complex<stan::math::var> &Complex<stan::math::var>::operator/=(stan::math::var x);

//binary operations
template<typename T>
std::ostream &operator<<(std::ostream &os, const Complex<T> &z)
{
    os << '(' << z.real() << ',' << z.imag() << ')';
}
template
std::ostream &operator<<(std::ostream &os, const Complex<double> &z);
template
std::ostream &operator<<(std::ostream &os, const Complex<stan::math::var> &z);

template<typename T>
bool operator==(Complex<T> LHS, Complex<T> RHS)
{
    return (LHS.real() == RHS.real() && LHS.imag() == RHS.imag());
}
template
bool operator==(Complex<double> LHS, Complex<double> RHS);
template
bool operator==(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
bool operator!=(Complex<T> LHS, Complex<T> RHS)
{
    return !(LHS == RHS);
}
template
bool operator!=(Complex<double> LHS, Complex<double> RHS);
template
bool operator!=(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator+(Complex<T> LHS, Complex<T> RHS)
{
    return LHS += RHS;
}
template
Complex<double> operator+(Complex<double> LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator+(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator-(Complex<T> LHS, Complex<T> RHS)
{
    return LHS -= RHS;
}
template
Complex<double> operator-(Complex<double> LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator-(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator-(Complex<T> LHS, T RHS)
{
    return LHS -= RHS;
}
template
Complex<double> operator-(Complex<double> LHS, double RHS);
template
Complex<stan::math::var> operator-(Complex<stan::math::var> LHS, stan::math::var RHS);

template<typename T>
Complex<T> operator-(T LHS, Complex<T> RHS)
{
    Complex<T> z(LHS);
    return z -= RHS;
}
template
Complex<double> operator-(double LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator-(stan::math::var LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator*(Complex<T> LHS, Complex<T> RHS)
{
    return LHS *= RHS;
}
template
Complex<double> operator*(Complex<double> LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator*(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator/(Complex<T> LHS, Complex<T> RHS)
{
    return LHS /= RHS;
}
template
Complex<double> operator/(Complex<double> LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator/(Complex<stan::math::var> LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator/(T LHS, Complex<T> RHS)
{
    Complex<T> z(LHS);
    return z /= RHS;
}
template
Complex<double> operator/(double LHS, Complex<double> RHS);
template
Complex<stan::math::var> operator/(stan::math::var LHS, Complex<stan::math::var> RHS);

template<typename T>
Complex<T> operator/(Complex<T> LHS, T RHS)
{
    return LHS /= RHS;
}
template
Complex<double> operator/(Complex<double> LHS, double RHS);
template
Complex<stan::math::var> operator/(Complex<stan::math::var> LHS, stan::math::var RHS);
