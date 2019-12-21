/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef DIIS_HEADER_H
#define DIIS_HEADER_H

#include <Eigen/Dense>

template<typename T>
class DIIS {
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
 public:
  MatrixXT prevVectors;
  MatrixXT errorVectors;
  MatrixXT diisMatrix;
  VectorXT bvector;
  int maxDim;
  int vectorDim;
  int iter;

  DIIS() {};

  DIIS(int pmaxDim, int pvectorDim) {
    init(pmaxDim, pvectorDim);
  }

  void init(int pmaxDim, int pvectorDim) {
    maxDim = pmaxDim; vectorDim = pvectorDim;

    prevVectors  = MatrixXT::Zero(pvectorDim, pmaxDim);
    errorVectors = MatrixXT::Zero(pvectorDim, pmaxDim);
    diisMatrix   = MatrixXT::Zero(pmaxDim+1, pmaxDim+1);
    bvector      = VectorXT::Zero(maxDim+1);
    iter = 0;
    for (int i=0; i<maxDim; i++) {
      diisMatrix(i, maxDim) = -1.0;
      diisMatrix(maxDim, i) = -1.0;
    }
    bvector(maxDim) = -1.;
  }

  void restart() {
    init(maxDim, vectorDim);
  }

  void update(VectorXT& newV, VectorXT& errorV) {

    prevVectors .col(iter%maxDim)  = newV;
    errorVectors.col(iter%maxDim)  = errorV;

    int col = iter%maxDim;
    for (int i=0; i<maxDim; i++) {
      diisMatrix(i   , col) = errorV.transpose()*errorVectors.col(i);
      diisMatrix(col, i   ) = diisMatrix(i, col);
    }
    iter++;

    if (iter < maxDim) {
      VectorXT b = VectorXT::Zero(iter+1);
      b[iter] = -1.0;
      MatrixXT localdiis = diisMatrix.block(0,0,iter+1, iter+1);
      for (int i=0; i<iter; i++) {
        localdiis(i, iter) = -1.0;
        localdiis(iter, i) = -1.0;
      }
      VectorXT x = localdiis.colPivHouseholderQr().solve(b);
      newV = prevVectors.block(0,0,vectorDim, iter)*x.head(iter);
      //+ errorVectors.block(0,0,vectorDim,iter)*x.head(iter);
    }
    else {
      VectorXT x = diisMatrix.colPivHouseholderQr().solve(bvector);
      newV = prevVectors*x.head(maxDim);// + errorVectors*x.head(maxDim);
      //if (iter == 20) {
      //iter = 0;
      //}
      //prevVectors.col((iter-1)%maxDim) = 1.* newV;
    }
  }
};

#endif
