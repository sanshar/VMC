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
#ifndef rWalkerHelper_HEADER_H
#define rWalkerHelper_HEADER_H

#include "rDeterminants.h"
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "ShermanMorrisonWoodbury.h"
#include "Slater.h"
#include "rJastrow.h"

template<typename Reference>
class rWalkerHelper {
};

template<>
class rWalkerHelper<Slater>
{

 public:
  HartreeFock hftype;                           //hftype same as that in slater
  std::array<MatrixXd, 2> thetaInv;          //inverse of the theta matrix
  vector<std::array<double, 2>> thetaDet;    //determinant of the theta matrix, vector for multidet

  vector<double> aoValues;                   //this is used to store the ao values at some coordinate
  std::array<MatrixXd, 2> DetMatrix;         //this is used to store the old determinant matrix
  std::array<MatrixXd, 2> Laplacian;         //each matrix L(elec, mo)
  std::vector<MatrixXd>   Gradient;          //each matrix G(3, mo) and there are nelec number of these
  rWalkerHelper() {};

  rWalkerHelper(const Slater &w, const rDeterminant &d) ;

  void initInvDetsTables(const Slater& w, const rDeterminant &d);


  void initInvDetsTablesGhf(const Slater& w, const rDeterminant &d);

  double getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d,
                      const Slater& w);

  double getDetFactorGHF(int i, Vector3d& newCoord,
                         int sz, int nelec, const Slater& w);
  
  double getDetFactor(int i, Vector3d& newCoord,
                      int sz, int nelec, const Slater& w);

  void updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d,
                    const Slater& w);


  void updateWalkerGHF(int elec, Vector3d& oldCoord, const rDeterminant &d,
                       int sz, int nelec, const Slater& w);
  
  
  void updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d,
                    int sz, int nelec, const Slater& w);

  void OverlapWithGradient(const rDeterminant& d, 
                           const Slater& w,
                           Eigen::VectorBlock<VectorXd>& grad,
                           const double& ovlp) ;

  void OverlapWithGradient(const rDeterminant& d, 
                           const Slater& w,
                           Eigen::VectorBlock<VectorXd>& grad);
  
  void OverlapWithGradientGhf(const rDeterminant& d, 
                           const Slater& w,
                           Eigen::VectorBlock<VectorXd>& grad) ;
  
};



template<>
class rWalkerHelper<rJastrow>
{
 public:
  //keep this updated
  double exponential;   //exponential due to all Jastrow Terms

  //Equation 33 of  https://doi.org/10.1063/1.4948778
  MatrixXd GradRatio; //nelec x 3 
  VectorXd LaplaceRatioIntermediate;
  VectorXd LaplaceRatio;
  
  rWalkerHelper() {};
  rWalkerHelper(const rJastrow& cps, const rDeterminant& d,
                MatrixXd& Rij, MatrixXd& RiN);

  void InitializeGradAndLaplaceRatio(const rJastrow& cps, const rDeterminant& d,
                                     MatrixXd& Rij, MatrixXd& RiN);
  
  //Assumes that Rij has already been updated
  void updateGradAndLaplaceRatio(int elec, Vector3d& oldCoord,
                                 const rJastrow& cps, const rDeterminant& d,
                                 MatrixXd& Rij, MatrixXd& RiN);

  void updateWalker(int i, Vector3d& oldcoord,
                    const rJastrow& cps, const rDeterminant& d,
                    MatrixXd& Rij, MatrixXd& RiN);


  //the position of the ith electron has changed
  double OverlapRatio(int i, Vector3d& coord, const rJastrow& cps,
                      const rDeterminant &d) const;


  void OverlapWithGradient(const rJastrow& cps,
                           VectorXd& grad,
                           const double& ovlp) const;


};  


#endif
