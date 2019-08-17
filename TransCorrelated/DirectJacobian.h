/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <vector>
#include <math.h>

using namespace std;
using namespace Eigen;

template<typename Scalar> struct DirectJacobian;
class Residuals;

namespace Eigen {
namespace internal {

template<>
struct traits<DirectJacobian<double>> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
{};

template<>
struct traits<DirectJacobian<complex<double>>> :  public Eigen::internal::traits<Eigen::SparseMatrix<complex<double> > >
{};
}
}

template<typename Scalar>
struct DirectJacobian : public Eigen::EigenBase<DirectJacobian<Scalar> > {

  using FunctorEvalResidue = boost::function<double (const VectorXd&, VectorXd&)>;
  typedef Scalar Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  
  typedef Matrix<Scalar, Dynamic, 1> FVectorType;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  
  vector<int> colIndex;
  vector<FVectorType> JacCols;
  FunctorEvalResidue& func;
  Scalar eps;
  FVectorType fvec;
  FVectorType xvec;
  
  //here it assumes that the Jaccol has been initialized
  Index rows() const { return fvec.size(); }
  Index cols() const { return fvec.size(); }

  
  DirectJacobian(
      FunctorEvalResidue& _func,
      Scalar epsfcn=1.e-4) : func(_func)
  {
    const Scalar epsmch = NumTraits<Scalar>::epsilon();
    eps = sqrt((std::max)(epsfcn,epsmch));
  };

  void setFvec(FVectorType& _xvec,
               FVectorType& _fvec) {
    xvec = _xvec;
    fvec = _fvec;
  }
  
  int PopulateJacobian(
      FVectorType& x,
      FVectorType& fvec)
  {
    int n = x.size();
    JacCols.resize(n, x); 


    /* computation of dense approximate jacobian. */
    for (int j = 0; j < n; ++j) {
      colIndex.push_back(j);
      
      Scalar temp = x[j];
      Scalar h = eps * abs(temp);
      if (h == 0.)
        h = eps;
      x[j] = temp + h;
      int iflag = func(x, JacCols[j]);
      if (iflag < 0)
        return iflag;
      x[j] = temp;
      JacCols[j] -= fvec;
      JacCols[j] /= h;
    }
    return 0;
  }

  void colWiseNorm(FVectorType& wa2) {
    for (int i=0; i<JacCols.size(); i++)
      wa2(i) = JacCols[i].blueNorm();
    return;
  }

  template<typename Rhs>
  Eigen::Product<DirectJacobian<Scalar>, Rhs, Eigen::AliasFreeProduct> operator*(
      const MatrixBase<Rhs>& x) const {
    
    return Eigen::Product<DirectJacobian<Scalar>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }
  
};
 
namespace Eigen {
namespace internal {

template<typename T, typename Rhs>
struct generic_product_impl<DirectJacobian<T>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<DirectJacobian<T>,Rhs,generic_product_impl<DirectJacobian<T>,Rhs> >
{
  typedef typename Product<DirectJacobian<T>,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst,
                            const DirectJacobian<T>& lhs,
                            const Rhs& rhs,
                            const Scalar& alpha)
  {
    double eps = std::sqrt(1 + lhs.xvec.norm())*1.5e-6/ rhs.norm();
    typename DirectJacobian<T>::FVectorType xplusu = lhs.xvec + eps * rhs;
    lhs.func(xplusu, dst);
    dst = (dst - lhs.fvec)/eps;

    //for(Index i=0; i<lhs.cols(); ++i)
    //dst += rhs(i) * lhs.JacCols[i];
  }
};
}
}



