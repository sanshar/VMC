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

struct GeneralTerm {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
  }
  
 public:
  virtual double exponential(const MatrixXd& Rij, const MatrixXd& RiN) const = 0;

  virtual double exponentDiff(int i, const Vector3d &coord,
                              const rDeterminant &d) = 0;
  
  //J = exp( sum_ij uij)
  //\sum_j Nabla^2_i uij  
  virtual void InitGradient(MatrixXd& Gradient,
                            const MatrixXd& rij,
                            const MatrixXd& RiN, const rDeterminant& d) = 0;
  virtual void InitLaplacian(VectorXd &laplacian,
                             const MatrixXd& rij,
                             const MatrixXd& RiN,
                             const rDeterminant& d) = 0;
  virtual void UpdateGradient(MatrixXd& Gradient,
                              const MatrixXd& rij,
                              const MatrixXd& RiN,
                              const rDeterminant& d,
                              const Vector3d& oldCoord,
                              int elecI) = 0;

  virtual void UpdateLaplacian(VectorXd& Laplacian,
                               const MatrixXd& rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord,
                               int elecI) = 0;


  virtual long getNumVariables() = 0;
  virtual void getVariables(Eigen::VectorXd& v, int& numVars) = 0;
  virtual void updateVariables(const Eigen::VectorXd& v, int &numVars) = 0;
  virtual void OverlapWithGradient(const MatrixXd& rij, const MatrixXd& RiN,
                                   const rDeterminant& d, VectorXd& grad,
                                   const double& ovlp, int& index) = 0;
  virtual void printVariables() = 0;
};


struct EEJastrow : public GeneralTerm{
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & boost::serialization::base_object<GeneralTerm>(*this);
    ar & alpha;
  }
  double alpha;
  double beta;
  
 public:
  EEJastrow();
  
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
};

struct ENJastrow : public GeneralTerm{
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & boost::serialization::base_object<GeneralTerm>(*this);
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
};


#endif
