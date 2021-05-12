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
#include "rMultiSlater.h"
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
  MatrixXd AO;                              //ne X Ao matrix -> ao_j(r_i)

  rWalkerHelper() {};

  rWalkerHelper(const rSlater &w, const rDeterminant &d);

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
                           const double& ovlp);

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
class rWalkerHelper<rMultiSlater>
{

 public:
  std::array<MatrixXcd, 2> A;         //reference configuration overlap matrix
  std::array<MatrixXcd, 2> Ainv;          //inverse of the reference configuration overlap matrix
  std::array<std::complex<double>, 2> detA;    //determinant of the reference configuration overlap matrix, D0
  std::array<MatrixXcd, 2> Abar;        //overlap matrix extended to unoccupied orbitals of the reference configuration

  std::array<MatrixXcd, 2> AinvAbar;         //matrix of single excitations: Ainv * Abar

  std::complex<double> totalRatio;                      //total overlap ratio with reference det = c0 + \sum_I cI * DI / D0
  std::array<Eigen::MatrixXcd, 2> Y; //Y (J. Chem. Theory Comput. 2017, 13, 5273−5281)

  mutable vector<double> aoValues;                   //this is used to store the ao values at some coordinate

  std::array<MatrixXcd, 2> Lap;          //lap A(elec, mo)
  std::array<std::array<MatrixXcd, 3>, 2> Grad;   //each of three matrices is grad A(elec, mo) 

  std::array<MatrixXcd, 2> Lapbar;          //lap Abar(elec, mo)
  std::array<std::array<MatrixXcd, 3>, 2> Gradbar;   //each of three matrices is grad Abar(elec, mo) 

  std::array<MatrixXd, 2> AOLap;                      //ne X Ao matrix -> Del^2_i ao_j(r_i)
  std::array<std::array<MatrixXd, 3>, 2>  AOGrad;        //ne X Ao matrix -> Del_ia ao_j(r_i), a=x,y,z
  std::array<MatrixXd, 2> AO;                              //ne X Ao matrix -> ao_j(r_i)

  rWalkerHelper() {};

  rWalkerHelper(const rMultiSlater &w, const rDeterminant &d);

  void initInvDetsTables(const rMultiSlater& w, const rDeterminant &d);

  double getDetFactor(int elec, Vector3d& newCoord, const rDeterminant &d, const rMultiSlater& w) const;
  
  void updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d, const rMultiSlater& w);
  
  void OverlapWithGradient(const rDeterminant& d, const rMultiSlater& w, Eigen::VectorBlock<VectorXd>& grad) const;
  
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
  
  //these are populated at initialization and updated with the walker
  VectorXd jastrowParams;
  VectorXd ParamValues;
  double exponential;

  //these are only populated when the GradientAndLaplacian() member function is called
  MatrixXd GradRatio; //nelec x 3 
  VectorXd LaplaceRatio;
  VectorXd LaplaceRatioIntermediate;

  MatrixXd ParamLaplacian;                      //nelec X njastrow matrix -> Del^2_i J_j
  std::array<MatrixXd, 3> ParamGradient;         //vector of GradRatio for each Jastrow

  //for four body jastrow
  //N, n, gradn, and lapn are populated at initialization and updated
  VectorXd N; //Vector of N_I
  MatrixXd n; //matrix of n_I (r_i)
  std::array<MatrixXd, 3> gradn;
  MatrixXd lapn;

  
  rWalkerHelper() {};
  rWalkerHelper(const rJastrow& cps, const rDeterminant& d,
                MatrixXd& Rij, MatrixXd& RiN);

  void updateWalker(int i, Vector3d& oldcoord,
                    const rJastrow& cps, const rDeterminant& d,
                    MatrixXd& Rij, MatrixXd& RiN);

  //the position of the ith electron has changed
  double OverlapRatio(int i, Vector3d& coord, const rJastrow& cps,
                      const rDeterminant &d) const;

  double OverlapRatioAndParamGradient(int i, Vector3d& coord, const rJastrow& cps, const rDeterminant &d, VectorXd &paramValues) const;

  void OverlapWithGradient(const rJastrow& cps,
                           VectorXd& grad,
                           const rDeterminant& d,
                           const double& ovlp) const;

  //populates param gradient and laplacian
  void GradientAndLaplacian(const rDeterminant &d);

  //returns vector of grad ratio for a specific electron, used in making moves
  void Gradient(int elec, Vector3d &gradRatio, const rDeterminant &d);

};  


#endif
