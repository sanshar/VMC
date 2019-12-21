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

template<typename Scalar, typename Functor> struct ParallelJacobian;
class Residuals;

namespace Eigen {
namespace internal {

template<typename F>
struct traits<ParallelJacobian<double, F>> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
{};

template<typename F>
struct traits<ParallelJacobian<complex<double>,F>> :  public Eigen::internal::traits<Eigen::SparseMatrix<complex<double> > >
{};
}
}

template<typename Scalar, typename Functor>
struct ParallelJacobian : public Eigen::EigenBase<ParallelJacobian<Scalar,Functor> > {

  //using FunctorEvalResidue = boost::function<double (const VectorXd&, VectorXd&)>;
  typedef Scalar Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  
  typedef Matrix<Scalar, Dynamic, 1> FVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> FMatrixType;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  
  vector<int> colIndex;
  FMatrixType Jac;
  Functor& func;
  Scalar eps;
  int nrows;
  int ncols;
  
  //here it assumes that the Jaccol has been initialized
  Index rows() const { return nrows; }
  Index cols() const { return ncols; }

  
  ParallelJacobian(
      Functor& _func,
      int _nrows,
      int _ncols,
      Scalar epsfcn=1.e-4) : func(_func), nrows(_nrows), ncols(_ncols)
  {
    const Scalar epsmch = NumTraits<Scalar>::epsilon();
    eps = sqrt((std::max)(epsfcn,epsmch));
  };

  int multiplyWithTranspose(FVectorType& x,
                            FVectorType& dst) {
    int colIndex = 0;
    for (int i=commrank; i<cols(); i+=commsize) {
      dst[i] = Jac.col(colIndex).transpose()*x;
      colIndex++;
    }
    int dstsize = dst.rows();
    MPI_Allreduce(MPI_IN_PLACE, &dst(0), dstsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  }
  
  int PopulateJacobian(
      FVectorType& x)
  {
    int n = x.size();
    int colPerProc = n/commsize + 1;
    
    Jac.resize(n, colPerProc); 
    FVectorType fvecTmp = x, fvec = x; fvec.setZero();

    func(x, fvec, true); //run in parallel

    /* computation of dense approximate jacobian. */
    int colIndex = 0;
    for (int j = commrank; j < n; j+=commsize) {

      Scalar temp = x[j];
      Scalar h = eps * abs(temp);
      if (h == 0.)
        h = eps;
      x[j] = temp + h;
      fvecTmp.setZero();
      func(x, fvecTmp, false);

      x[j] = temp;
      Jac.col(colIndex) = (fvecTmp - fvec)/h;
      colIndex++;
    }
    return 0;
  }

  //void colWiseNorm(FVectorType& wa2) {
  //for (int i=0; i<JacCols.size(); i++)
  //wa2(i) = JacCols[i].blueNorm();
  //return;
  //}

  template<typename Rhs>
  Eigen::Product<ParallelJacobian<Scalar, Functor>, Rhs, Eigen::AliasFreeProduct> operator*(
      const MatrixBase<Rhs>& x) const {
    
    return Eigen::Product<ParallelJacobian<Scalar, Functor>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }
  
};
 
namespace Eigen {
namespace internal {

template<typename T, typename F, typename Rhs>
struct generic_product_impl<ParallelJacobian<T,F>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<ParallelJacobian<T,F>,Rhs,generic_product_impl<ParallelJacobian<T,F>,Rhs> >
{
  typedef typename Product<ParallelJacobian<T,F>,Rhs>::Scalar Scalar;

  //J'J*v
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst,
                            const ParallelJacobian<T,F>& lhs,
                            const Rhs& rhs,
                            const Scalar& alpha)
  {
    typename ParallelJacobian<T,F>::FVectorType xtemp(lhs.rows()); xtemp.setZero();

    //multiply x with J
    int colIndex = 0;
    for (int i=commrank; i<lhs.cols(); i+=commsize) {
      xtemp += lhs.Jac.col(colIndex)*rhs(i);
      colIndex++;
    }
    int xsize = xtemp.rows();
    MPI_Allreduce(MPI_IN_PLACE, &xtemp(0), xsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    //multiply xtemp with J'
    colIndex = 0;
    for (int i=commrank; i<lhs.cols(); i+=commsize) {
      dst[i] = dst[i] + alpha*lhs.Jac.col(colIndex).transpose()*xtemp;
      colIndex++;
    }
    int dstsize = dst.rows();
    MPI_Allreduce(MPI_IN_PLACE, &dst(0), dstsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    dst.noalias() += 1.e-6*rhs;
  }
};
}
}



