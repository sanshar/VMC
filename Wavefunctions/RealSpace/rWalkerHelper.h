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
#include "rSlater.h"
#include "rBFSlater.h"
#include "rJastrow.h"
#include "JastrowTermsHardCoded.h"

template<typename Reference>
class rWalkerHelper {
};

template<>
class rWalkerHelper<rSlater>
{

 public:
  HartreeFock hftype;                           //hftype same as that in slater
  std::array<MatrixXcd, 2> thetaInv;          //inverse of the theta matrix
  vector<std::array<std::complex<double>, 2>> thetaDet;    //determinant of the theta matrix, vector for multidet

  mutable vector<double> aoValues;                   //this is used to store the ao values at some coordinate
  std::array<MatrixXcd, 2> DetMatrix;         //this is used to store the old determinant matrix
  MatrixXcd Laplacian;                          //L(elec, mo)
  std::array<MatrixXcd, 3>   Gradient;        //each of three matrices is G(elec, mo) 
  MatrixXd AOLaplacian;                      //ne X Ao matrix -> Del^2_i ao_j(r_i)
  std::array<MatrixXd,3>  AOGradient;        //ne X Ao matrix -> Del_ia ao_j(r_i), a=x,y,z
  
  rWalkerHelper() {};

  rWalkerHelper(const rSlater &w, const rDeterminant &d) ;

  void initInvDetsTables(const rSlater& w, const rDeterminant &d);


  void initInvDetsTablesGhf(const rSlater& w, const rDeterminant &d);

  double getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d, const rSlater& w) const;

  double getDetFactorGHF(int i, Vector3d& newCoord, int sz, int nelec, const rSlater& w) const;
  
  double getDetFactor(int i, Vector3d& newCoord, int sz, int nelec, const rSlater& w) const;

  void updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d,
                    const rSlater& w);


  void updateWalkerGHF(int elec, Vector3d& oldCoord, const rDeterminant &d,
                       int sz, int nelec, const rSlater& w);
  
  
  void updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d,
                    int sz, int nelec, const rSlater& w);

  void OverlapWithGradient(const rDeterminant& d, 
                           const rSlater& w,
                           Eigen::VectorBlock<VectorXd>& grad,
                           const double& ovlp) ;

  void HamOverlap(const rDeterminant& d, 
                  const rSlater& w,
                  MatrixXd& Rij, MatrixXd& RiN,
                  Eigen::VectorBlock<VectorXd>& hamgrad);
  
  void OverlapWithGradient(const rDeterminant& d, 
                           const rSlater& w,
                           Eigen::VectorBlock<VectorXd>& grad);
  
  void OverlapWithGradientGhf(const rDeterminant& d, 
                           const rSlater& w,
                           Eigen::VectorBlock<VectorXd>& grad) ;
  
};

template<>
class rWalkerHelper<rBFSlater>
{

 public:
  rDeterminant dp;                             //backflow displaced positions
  rDeterminant proposedDp;                     //backflow displaced positions for the proposed move
  vector<Vector3d> displacements;              //backflow displacements without h factors
  vector<std::array<Vector3d, 2>> gradbDisplacements; //sum_j (r_{i\mu} - r_{j\mu}) 2 r_{ij}^2 eta_0(r_{ij}) / b^3, opposite and same spin, helper for gradient
  vector<std::array<Vector3d, 2>> gradaDisplacements; //sum_j (r_{i\mu} - r_{j\mu}) eta_0(r_{ij}), opposite and same spin, helper for gradient
  vector<Vector3d> gradbNDisplacements; //sum_I (r_{i\mu} - r_{I\mu}) 2 r_{iI}^2 chi(r_{iI}) / b^3, helper for gradient
  vector<Vector3d> gradaNDisplacements; //sum_I (r_{i\mu} - r_{I\mu}) chi(r_{iI}), helper for gradient
  mutable vector<double> aoValues;             //this is used to store the ao values and derivatives at some coordinate
  MatrixXd etaValues;                          //eta(r_{ij})
  MatrixXd chiValues;                          //chi(r_{iI})
  VectorXd hValues;                            //h(r_i) = \Pi_I g(r_{iI})
  std::array<VectorXd, 3> hGradient;           //dh(r_i) / dr_{i\mu}
  std::array<VectorXd, 3> hSecondDerivatives;  //d^2h(r_i) / dr_{i\mu}^2
  std::array<MatrixXd, 9> rpGradient;          //rp=r', dr'_{i\mu} / dr_{j\nu}
  std::array<MatrixXd, 9> rpSecondDerivatives; //rp=r', d^2r'_{i\mu} / dr_{j\nu}^2
  std::complex<double> thetaDet;               //determinant of the theta matrix, vector for multidet
  std::complex<double> proposedThetaDet;       //determinant of the proposed move
  MatrixXcd DetMatrix;                         //this is used to store the current determinant matrix
  MatrixXcd proposedDetMatrix;                 //this is used to store the determinant matrix for the proposed move
  MatrixXcd thetaInv;                          //inverse of the theta matrix
  std::array<MatrixXcd, 3> rTable;             //gradient * thetaInv
  std::array<MatrixXcd, 6> MOSecondDerivatives;//second derivatives of mo's w.r.t. rp, order: xx, xy, xz, yy, yz, zz
  std::array<MatrixXcd, 3> MOGradient;         //each of three matrices is G(elec, mo) 
  VectorXcd slaterLaplacianRatio;              //w.r.t. r_i
  std::array<VectorXcd, 3> slaterGradientRatio;//w.r.t. r_i

  rWalkerHelper() {};
  rWalkerHelper(const rBFSlater &w, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN);

  void initPositionTables(const rBFSlater &w, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN);
  
  void calcDetMatrix(const rBFSlater& w, const rDeterminant &d);
  void calcSlaterDerivatives(const rBFSlater& w, const rDeterminant &d);
  
  //void initInvDetsTables(const rBFSlater& w, const rDeterminant &d);

  double getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d, const rBFSlater& w);

  void updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN,
                    const rBFSlater& w);

  void OverlapWithGradient(const rDeterminant& d, 
                           const rBFSlater& w,
                           Eigen::VectorBlock<VectorXd>& grad,
                           const double& ovlp) ;

  void HamOverlap(const rDeterminant& d, 
                  const rBFSlater& w,
                  MatrixXd& Rij, MatrixXd& RiN,
                  Eigen::VectorBlock<VectorXd>& hamgrad);
  
  void OverlapWithGradient(const rDeterminant& d, 
                           const rBFSlater& w,
                           Eigen::VectorBlock<VectorXd>& grad);
  
};


template<>
class rWalkerHelper<rJastrow>
{
 public:
  int Qmax;
  int QmaxEEN;
  int EEsameSpinIndex,
      EEoppositeSpinIndex,
      ENIndex,
      EENsameSpinIndex,
      EENoppositeSpinIndex,
      EENNlinearIndex,
      EENNIndex;
  
  //Equation 33 of  https://doi.org/10.1063/1.4948778
  double   exponential;
  MatrixXd GradRatio; //nelec x 3 
  VectorXd LaplaceRatioIntermediate;
  VectorXd LaplaceRatio;

  VectorXd ParamValues;
  MatrixXd ParamLaplacian;                      //nelec X njastrow matrix -> Del^2_i J_j
  std::vector<MatrixXd>  ParamGradient;         //vector of GradRatio for each Jastrow
  MatrixXd workMatrix;
  VectorXd jastrowParams;

  
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
                           const rDeterminant& d,
                           const double& ovlp) const;

  void HamOverlap(const rJastrow& cps,
                  VectorXd& grad,
                  const rDeterminant& d,
                  const double& ovlp) const;

};  


#endif
