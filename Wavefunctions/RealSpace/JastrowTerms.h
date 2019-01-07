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

#ifndef RJastrowTerms_HEADER_H
#define RJastrowTerms_HEADER_H


#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "rDeterminants.h"
#include "boost/serialization/export.hpp"
#include "global.h"
#include "input.h"

using namespace Eigen;
using namespace std;

struct GeneralJastrow {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & beta & I & J & m & n &o & ss & fixed ;
  }
 public:

  vector<int> I, J, m, n, o, ss, fixed; //this description is from equation 41 of JCP 133 (142109)
  //ss = 1 means same spin, and ss = 0 means opposite spin and ss=2 both spins
  //fixed=1 means it will not be optimized
  //fixed=0 means it will be optimized
  
  double beta;
  vector<Vector3d>& Ncoords;
  vector<double>&   Ncharge;
  
  GeneralJastrow();

  double getExpLaplaceGradIJ(int i, int j,
                             Vector3d& gi, Vector3d& gj,
                             double& laplaciani,
                             double& laplacianj,
                             const Vector3d& coordi,
                             const Vector3d& coordj,
                             const double* params,
                             double * gradHelper,
                             double factor,
                             bool dolaplaceGrad) const;

  double getExpLaplaceGradIJperJastrow(int i, int j,
                                       vector<MatrixXd>& gradient,
                                       MatrixXd& laplacian,
                                       const Vector3d& coordi,
                                       const Vector3d& coordj,
                                       const double* params,
                                       double * gradHelper,
                                       double factor,
                                       bool dolaplaceGrad) const;
  
  double exponentialInitLaplaceGrad(const rDeterminant& d,
                                    vector<MatrixXd>& paramGradient,
                                    MatrixXd& laplacian,
                                    const double * params,
                                    double * gradHelper) const;
  

  double exponentDiff(int i, const Vector3d &coord,
                      const rDeterminant &d,
                      const double * params,
                      double * values) const;

  void UpdateLaplaceGrad(vector<MatrixXd>& Gradient,
                         MatrixXd& laplacian,
                         const rDeterminant& d,
                         const Vector3d& oldCoord,
                         int i,
                         const double * params,
                         double * gradHelper) const;
  
  void OverlapWithGradient(VectorXd& grad, int& index,
                           const rDeterminant& d,
                           const vector<double>& params,
                           const vector<double>& values) const ;
  
};




#endif
