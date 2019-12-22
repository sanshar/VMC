#ifndef COMPLEX_HEADER_H
#define COMPLEX_HEADER_H
#include <stan/math.hpp>
#include <complex>

template<typename T>
class Complex
{
 private:
  T Real;
  T Imag;

 public:

  Complex(T _Real =0.0, T _Imag=0.0) : Real(_Real), Imag(_Imag) {};

  Complex(const Complex<T> &z)
  {
    Real = z.Real;
    Imag = z.Imag;
  }
  
  const T& real() const
  {
    return Real;
  }
  
  const T& imag() const
  {
    return Imag;
  }

  T& real()
  {
    return Real;
  }
  
  T& imag()
  {
    return Imag;
  }
  
  //setter
  void real(T _Real)
  {
    Real = _Real;
  }

  void imag(T _Imag)
  {
    Imag = _Imag;
  }

  //conjugate
  
  Complex<T> conj() const
  {
    Complex<T> zcopy(Real, -Imag);
    return zcopy;
  }

  //norm
  
  T squaredNorm() const
  {
    return Real * Real + Imag * Imag;
  }

  //unary operators
  
  Complex<T> &operator=(const Complex<T> &z)
  {
    Real = z.Real;
    Imag = z.Imag;
    return *this;
  }

  
  Complex<T> &operator=(const T &x)
  {
    Real = x;
    Imag = 0.0;
    return *this;
  }

  
  Complex<T> &operator=(const std::complex<double> &z)
  {
    Real = z.real();
    Imag = z.imag();
    return *this;
  }

  
  Complex<T> &operator+=(const Complex<T> &z)
  {
    Real += z.Real;
    Imag += z.Imag;
    return *this;
  }

  
  Complex<T> &operator-=(const Complex<T> &z)
  {
    Real -= z.Real;
    Imag -= z.Imag;
    return *this;
  }

  
  Complex<T> &operator-=(const T &x)
  {
    Real -= x;
    return *this;
  }

  
  Complex<T> &operator*=(const Complex<T> &z)
  {
    T a = Real, b = Imag, c = z.Real, d = z.Imag;
    Real = a * c - b * d;
    Imag = b * c + a * d;
    return *this;
  }

  
  Complex<T> &operator/=(const Complex<T> &z)
  {
    assert(z.Real || z.Imag);
    T a = Real, b = Imag, c = z.Real, d = z.Imag;
    T r2 = z.squaredNorm();
    Real = (a * c + b * d) / r2;
    Imag = (b * c - a * d) / r2;
    return *this;
  }

  
  Complex<T> &operator/=(const T &x)
  {
    Complex<T> z(x);
    return *this /= z;
  }
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
std::ostream &operator<<(std::ostream &os, const Complex<T> &z)
{
  os << '(' << z.real() << ',' << z.imag() << ')';
}

template<typename T>
bool operator==(const Complex<T> &LHS, const Complex<T> &RHS)
{
  return (LHS.real() == RHS.real() && LHS.imag() == RHS.imag());
}

template<typename T>
bool operator!=(const Complex<T> &LHS, const Complex<T> &RHS)
{
  return !(LHS == RHS);
}

template<typename T>
Complex<T> operator+(const Complex<T> &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy += RHS;
}

template<typename T>
Complex<T> operator-(const Complex<T> &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy -= RHS;
}

template<typename T>
Complex<T> operator-(const Complex<T> &LHS, const T &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy -= RHS;
}

template<typename T>
Complex<T> operator-(const T &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy -= RHS;
}

template<typename T>
Complex<T> operator*(const Complex<T> &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy *= RHS;
}

template<typename T>
Complex<T> operator*(const Complex<T> &LHS, const T &RHS)
{
  Complex<T> zcopy(LHS);
  zcopy.real() *= RHS; 
  zcopy.imag() *= RHS; 
  return zcopy;
}

template<typename T>
Complex<T> operator*(const T &RHS, const Complex<T> &LHS)
{
  Complex<T> zcopy(LHS);
  zcopy.real() *= RHS; 
  zcopy.imag() *= RHS; 
  return zcopy;
}



template<typename T>
Complex<T> operator/(const Complex<T> &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy /= RHS;
}

template<typename T>
Complex<T> operator/(const T &LHS, const Complex<T> &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy /= RHS;
}

template<typename T>
Complex<T> operator/(const Complex<T> &LHS, const T &RHS)
{
  Complex<T> zcopy(LHS);
  return zcopy /= RHS;
}


//Eigen compatability
template<typename T>
inline Complex<T> conj(const Complex<T> &z)  { return z.conj(); }

template<typename T>
inline T real(const Complex<T> &z)  { return z.real(); }

template<typename T>
inline T imag(const Complex<T> &z)    { return z.imag(); }

template<typename T>
inline T abs(const Complex<T> &z)  { return z.squaredNorm(); }

template<typename T>
inline T abs2(const Complex<T> &z)  { return z.squaredNorm(); }

template<typename T>
inline Complex<T> exp(const Complex<T> &z)  {
  Complex<T> zout(exp(z.real())*cos(z.imag()), exp(z.real())*sin(z.imag()));
  return zout;
}

//This is a bit ugly because we are having to make a special case of the function with double

Complex<stan::math::var> operator*(const Complex<stan::math::var> &LHS, const double &RHS);

Complex<stan::math::var> operator*(const double &RHS, const Complex<stan::math::var> &LHS);

//Complex<stan::math::var> operator/(const double &LHS, const Complex<stan::math::var> &RHS){
//Complex<stan::math::var> zout(LHS.real()*RHS, LHS.imag()*RHS);
//return zout;
//}

Complex<stan::math::var> operator/(const Complex<stan::math::var> &LHS, const double &RHS);



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
