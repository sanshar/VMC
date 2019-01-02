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

  double getExponentialIJ(int i, int j,
                          const Vector3d& coordi, const Vector3d& coordj,
                          const double* params,
                          double * gradHelper,
                          double factor) const;

  void getGradientIJ(Vector3d& gi, Vector3d& gj, 
                     int i, int j,
                     const Vector3d& coordi, const Vector3d& coordj,
                     const double* params) const;

  void getLaplacianIJ(double& laplaciani, double& laplacianj,
                      int i, int j,
                      const Vector3d& coordi, const Vector3d& coordj,
                      const double* params) const ;
  
  double exponential(const rDeterminant& d,
                     const double * params,
                     double * gradHelper) const;

  double exponentDiff(int i, const Vector3d &coord,
                      const rDeterminant &d,
                      const double * params,
                      double * values) const;

  void InitGradient(MatrixXd& Gradient,
                    const rDeterminant& d,
                    const double * params) const;

  void InitLaplacian(VectorXd &laplacian,
                     const rDeterminant& d,
                     const double * params) const;
  
  void UpdateGradient(MatrixXd& Gradient,
                      const rDeterminant& d,
                      const Vector3d& oldCoord,
                      int elecI, const double * params,
                      double* values) const;

  void UpdateLaplacian(VectorXd &laplacian,
                       const rDeterminant& d,
                       const Vector3d& oldCoord,
                       int elecI,
                       const double * params) const;

  void OverlapWithGradient(VectorXd& grad, int& index,
                           const vector<double>& values) const ;
  
};



struct EEJastrow {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & beta;
  }
  double beta;
  
 public:
  EEJastrow();
  
  double exponential(const MatrixXd& rij, const MatrixXd& RiN,
                     int maxOrder, const double * params,
                     double * values) const;

  double exponentDiff(int i, const Vector3d &coord,
                      const rDeterminant &d,
                      int maxOrder, const double * params,
                      double * values) const;

  void InitGradient(MatrixXd& Gradient,
                    const MatrixXd& rij,
                    const MatrixXd& RiN,
                    const rDeterminant& d,
                    int maxOrder, const double * params) const;

  void InitLaplacian(VectorXd &laplacian,
                     const MatrixXd& rij,
                     const MatrixXd& RiN,
                     const rDeterminant& d,
                     int maxOrder, const double * params) const;
  
  void UpdateGradient(MatrixXd& Gradient,
                      const MatrixXd& rij,
                      const MatrixXd& RiN,
                      const rDeterminant& d,
                      const Vector3d& oldCoord,
                      int elecI, int maxOrder, const double * params,
                      double* values) const;

  void UpdateLaplacian(VectorXd &laplacian,
                       const MatrixXd& rij,
                       const MatrixXd& RiN,
                       const rDeterminant& d,
                       const Vector3d& oldCoord,
                       int elecI,
                       int maxOrder, const double * params) const;

  void OverlapWithGradient(VectorXd& grad, int& index,
                           int minOrder, int maxOrder,
                           const double * values) const ;
  
};


struct ENJastrow {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & beta
        & Ncoords
        & Ncharge;
  }
  vector<Vector3d> Ncoords;
  vector<double>   Ncharge;
  double beta;
  
 public:
  ENJastrow();
  
  double exponential(const MatrixXd& rij, const MatrixXd& RiN,
                     int maxOrder, const double * params,
                     double * values) const;

  double exponentDiff(int i, const Vector3d &coord,
                      const rDeterminant &d,
                      int maxOrder, const double * params,
                      double * values) const;

  void InitGradient(MatrixXd& Gradient,
                    const MatrixXd& rij,
                    const MatrixXd& RiN,
                    const rDeterminant& d,
                    int maxOrder, const double * params) const;

  void InitLaplacian(VectorXd &laplacian,
                     const MatrixXd& rij,
                     const MatrixXd& RiN,
                     const rDeterminant& d,
                     int maxOrder, const double * params) const;
  
  void UpdateGradient(MatrixXd& Gradient,
                      const MatrixXd& rij,
                      const MatrixXd& RiN,
                      const rDeterminant& d,
                      const Vector3d& oldCoord,
                      int elecI, int maxOrder, const double * params,
                      double* values) const;

  void UpdateLaplacian(VectorXd &laplacian,
                       const MatrixXd& rij,
                       const MatrixXd& RiN,
                       const rDeterminant& d,
                       const Vector3d& oldCoord,
                       int elecI,
                       int maxOrder, const double * params) const;

  void OverlapWithGradient(VectorXd& grad, int& index,
                           int minOrder, int maxOrder,
                           const double * values) const ;
  
};
/*
struct ENJastrow {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & Ncoords;
    ar & Ncharge;
    ar & alpha;
  }
  vector<Vector3d> Ncoords;
  vector<double>   Ncharge;
  vector<double> alpha;
  
 public:
  ENJastrow() ;
  double exponential(const MatrixXd& rij, const MatrixXd& RiN) const;

  double exponentDiff(int i, const Vector3d &coord,
                      const rDeterminant &d);

  void InitGradient(MatrixXd& Gradient,
                    const MatrixXd& rij,
                    const MatrixXd& RiN,
                    const rDeterminant& d);

  //J = exp( sum_ij uij)
  //\sum_j Nabla^2_i uij  
  void InitLaplacian(VectorXd &laplacian,
                     const MatrixXd& rij,
                     const MatrixXd& RiN,
                     const rDeterminant& d);
  
  //J = exp( sum_ij uij)
  //\sum_j Nabla^2_i uij  
  void UpdateGradient(MatrixXd& Gradient,
                      const MatrixXd& rij,
                      const MatrixXd& RiN,
                      const rDeterminant& d,
                      const Vector3d& oldCoord,
                      int elecI);

  //J = exp( sum_ij uij)
  //\sum_j Nabla^2_i uij  
  void UpdateLaplacian(VectorXd &laplacian,
                       const MatrixXd& rij,
                       const MatrixXd& RiN,
                       const rDeterminant& d,
                       const Vector3d& oldCoord,
                       int elecI);

  long getNumVariables();

  void getVariables(Eigen::VectorXd& v, int& numVars) ;
  
  void updateVariables(const Eigen::VectorXd& v, int &numVars);

  void OverlapWithGradient(const MatrixXd& rij, const MatrixXd& RiN,
                           const rDeterminant& d, VectorXd& grad,
                           const double& ovlp, int& index);
  
  void printVariables();
  };*/


#endif
