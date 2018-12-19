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

  rWalkerHelper(const Slater &w, const rDeterminant &d) 
  {
    hftype = w.hftype;
 
    //fill the spin strings for the walker and the zeroth reference det
    thetaDet.resize(w.getNumOfDets());

    if (hftype == Generalized) {
      initInvDetsTablesGhf(w, d);
    }
    else {
      initInvDetsTables(w, d);
    }
  }

  void initInvDetsTables(const Slater& w, const rDeterminant &d) {
    int norbs = Determinant::norbs;
    aoValues.resize(10*norbs, 0.0);

    DetMatrix[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
    DetMatrix[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

    Gradient.resize(d.nalpha, MatrixXd::Zero(3, d.nalpha));
    Gradient.resize(d.nalpha+d.nbeta, MatrixXd::Zero(3, d.nbeta));
    Laplacian[0] = MatrixXd::Zero(d.nalpha, d.nalpha);

    for (int elec=0; elec<d.nalpha; elec++) {
      schd.gBasis.eval_deriv2(d.coord[elec], &aoValues[0]);

      for (int mo=0; mo<d.nalpha; mo++) 
        for (int j=0; j<norbs; j++) {
          DetMatrix[0](elec, mo) += aoValues[j] * w.getHforbs(0)(j, mo);

          Laplacian[0](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] ) * w.getHforbs(0)(j,mo);

          Gradient[elec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(0)(j, mo);
          Gradient[elec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(0)(j, mo);
          Gradient[elec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(0)(j, mo);
        }          
    }
    
    Eigen::FullPivLU<MatrixXd> lua(DetMatrix[0]);
    if (lua.isInvertible()) {
      thetaInv[0] = lua.inverse();
      thetaDet[0][0] = lua.determinant();
    }
    else {
      cout << " overlap with alpha determinant not invertible" << endl;
      exit(0);
    }
    cout << Laplacian[0]<<" ld  "<<DetMatrix[0]<<endl;
    
    Laplacian[1] = MatrixXd::Zero(d.nbeta, d.nbeta);
    for (int elec=0; elec<d.nbeta; elec++) {
      schd.gBasis.eval_deriv2(d.coord[elec+d.nalpha], &aoValues[0]);
      for (int mo=0; mo<d.nbeta; mo++) 
        for (int j=0; j<norbs; j++) {
          DetMatrix[1](elec, mo) += aoValues[j] * w.getHforbs(1)(j, mo);

          Laplacian[1](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] ) * w.getHforbs(1)(j,mo);
      
          Gradient[d.nalpha + elec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(1)(j, mo);
          Gradient[d.nalpha + elec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(1)(j, mo);
          Gradient[d.nalpha + elec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(1)(j, mo);
        }
    }

    if (d.nbeta != 0) {
      Eigen::FullPivLU<MatrixXd> lub(DetMatrix[1]);
      if (lub.isInvertible()) {
        thetaInv[1] = lub.inverse();
        thetaDet[0][1] = lub.determinant();
      }
      else {
        cout << " overlap with beta determinant not invertible" << endl;
        exit(0);
      }
    }

    
  }


  void initInvDetsTablesGhf(const Slater& w, const rDeterminant &d) {
    int norbs = Determinant::norbs;
    //aoValues.resize(10*norbs);

    DetMatrix[0] = MatrixXd::Zero(d.nelec, d.nelec);
    Gradient.resize(d.nelec, MatrixXd::Zero(3, d.nelec));
    Laplacian[0] = MatrixXd::Zero(d.nelec, d.nelec);

    
    for (int elec=0; elec<d.nelec; elec++) {
      aoValues.resize(10*norbs, 0.0);
      schd.gBasis.eval_deriv2(d.coord[elec], &aoValues[0]);

      for (int mo=0; mo<d.nelec; mo++) 
        for (int j=0; j<norbs; j++) {
          DetMatrix[0](elec, mo) += aoValues[j] * w.getHforbs()(j, mo);

          Laplacian[0](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] ) * w.getHforbs(0)(j,mo);

          Gradient[elec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(0)(j, mo);
          Gradient[elec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(0)(j, mo);
          Gradient[elec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(0)(j, mo);
          
        }
    }
    
    Eigen::FullPivLU<MatrixXd> lu(DetMatrix[0]);
    if (lu.isInvertible()) {
      thetaInv[0] = lu.inverse();
      thetaDet[0][0] = lu.determinant();
    }
    else {
      cout << " overlap with GHF determinant not invertible" << endl;
      exit(0);
    }

  }

  double getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d,
                    const Slater& w) {
    if (hftype == Generalized) 
      return getDetFactor(i, newCoord, 0, d.nelec, w);
    else if (i < d.nalpha)
      return getDetFactor(i, newCoord, 0, d.nalpha, w);
    else
      return getDetFactor(i-d.nalpha, newCoord, 1, d.nbeta, w);
  }

  double getDetFactor(int i, Vector3d& newCoord,
                    int sz, int nelec, const Slater& w) {
    int norbs = Determinant::norbs;
    aoValues.resize(norbs);

    schd.gBasis.eval(newCoord, &aoValues[0]);


    VectorXd newVec = VectorXd::Zero(nelec);
    for (int mo=0; mo<nelec; mo++) 
      for (int j=0; j<norbs; j++) 
        newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);

    return (newVec.dot(thetaInv[sz].col(i)));
  }

  void updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d,
                    const Slater& w) {
    if (hftype == Generalized) 
      updateWalker(i, oldCoord, d, 0, d.nelec, w);
    else if (i < d.nalpha)
      updateWalker(i, oldCoord, d, 0, d.nalpha, w);
    else
      updateWalker(i-d.nalpha, oldCoord, d, 1, d.nbeta, w);

  }

  void updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d,
                    int sz, int nelec, const Slater& w) {
    int norbs = Determinant::norbs;
    aoValues.resize(10 * norbs, 0.0);

    int gelec = elec;
    if (sz == 1) gelec += d.nalpha;    

    schd.gBasis.eval_deriv2(d.coord[gelec], &aoValues[0]);

    VectorXd newVec = VectorXd::Zero(nelec);
    for (int mo=0; mo<nelec; mo++) 
      for (int j=0; j<norbs; j++) 
        newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);

    
    calculateInverseDeterminantWithRowChange(thetaInv[sz],thetaDet[0][sz],DetMatrix[sz],
                                             elec, newVec);

    for (int mo=0; mo<nelec; mo++) {
      Laplacian[sz](elec, mo) = 0.0;


      Gradient[gelec](0, mo) = 0.0;
      Gradient[gelec](1, mo) = 0.0;
      Gradient[gelec](2, mo) = 0.0;
      
      for (int j=0; j<norbs; j++) {
        Laplacian[sz](elec, mo) += (  aoValues[4*norbs+j]
                                      + aoValues[7*norbs+j]
                                      + aoValues[9*norbs+j] ) * w.getHforbs(sz)(j,mo);
        
        
        Gradient[gelec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(sz)(j, mo);
        Gradient[gelec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(sz)(j, mo);
        Gradient[gelec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(sz)(j, mo);
      }
      
    }

  }
};



template<>
class rWalkerHelper<rJastrow>
{
 public:
  //keep this updated
  double exponential;   //exponential due to all Jastrow Terms
  MatrixXd Rij;         //the inter-electron distances
  MatrixXd RiN;         //electron-nucleus distances

  //Equation 33 of  https://doi.org/10.1063/1.4948778
  MatrixXd GradRatio; //nelec x 3 
  VectorXd LaplaceRatioIntermediate;
  VectorXd LaplaceRatio;
  
  rWalkerHelper() {};
  rWalkerHelper(const rJastrow& cps, const rDeterminant& d) {

    //RIJ matrix
    Rij = MatrixXd::Zero(d.nelec, d.nelec);
    for (int i=0; i<d.nelec; i++)
      for (int j=0; j<i; j++) {
        double rij = pow( pow(d.coord[i][0] - d.coord[j][0], 2) +
                          pow(d.coord[i][1] - d.coord[j][1], 2) +
                          pow(d.coord[i][2] - d.coord[j][2], 2), 0.5);

        Rij(i,j) = rij;
        Rij(j,i) = rij;        
      }

    RiN = MatrixXd::Zero(d.nelec, schd.Ncoords.size());
    for (int i=0; i<d.nelec; i++)
      for (int j=0; j<schd.Ncoords.size(); j++) {
        double rij = pow( pow(d.coord[i][0] - schd.Ncoords[j][0], 2) +
                          pow(d.coord[i][1] - schd.Ncoords[j][1], 2) +
                          pow(d.coord[i][2] - schd.Ncoords[j][2], 2), 0.5);

        RiN(i,j) = rij;
      }
    
    //make exponential
    exponential = 0.0;
    for (int i=0; i<d.nelec; i++) 
      for (int t =0; t<cps.Terms.size(); t++)
        exponential += cps.Terms[t]->exponential(Rij, RiN);

    GradRatio = MatrixXd::Zero(d.nelec,3);
    LaplaceRatio.resize(d.nelec);
    LaplaceRatioIntermediate.resize(d.nelec);
    InitializeGradAndLaplaceRatio(cps, d);
  }


  void InitializeGradAndLaplaceRatio(const rJastrow& cps, const rDeterminant& d) {
    for (int t =0; t<cps.Terms.size(); t++) {
      cps.Terms[t]->InitGradient(GradRatio, Rij, RiN, d);
      cps.Terms[t]->InitLaplacian(LaplaceRatioIntermediate, Rij, RiN, d);
    }

    for (int i=0; i<d.nelec; i++) {
      LaplaceRatio[i] = LaplaceRatioIntermediate[i] +
          pow(GradRatio(i,0), 2) +
          pow(GradRatio(i,1), 2) +
          pow(GradRatio(i,2), 2) ;
    }
  }
  
  //Assumes that Rij has already been updated
  void updateGradAndLaplaceRatio(int elec, Vector3d& oldCoord,
                                 const rJastrow& cps, const rDeterminant& d) {

    for (int t =0; t<cps.Terms.size(); t++) {
      cps.Terms[t]->UpdateGradient(GradRatio, Rij, RiN, d, oldCoord, elec);
      cps.Terms[t]->UpdateLaplacian(LaplaceRatioIntermediate, Rij, RiN, d, oldCoord, elec);
    }

    for (int i=0; i<d.nelec; i++) {
      LaplaceRatio[i] = LaplaceRatioIntermediate[i] +
          pow(GradRatio(i,0), 2) +
          pow(GradRatio(i,1), 2) +
          pow(GradRatio(i,2), 2) ;
    }
  }

  void updateWalker(int i, Vector3d& oldcoord,
                    const rJastrow& cps, const rDeterminant& d) {
    for (int j=0; j<d.nelec; j++) {
      Rij(i, j) = pow( pow(d.coord[i][0] - d.coord[j][0], 2) +
                       pow(d.coord[i][1] - d.coord[j][1], 2) +
                       pow(d.coord[i][2] - d.coord[j][2], 2), 0.5);

      Rij(j,i) = Rij(i,j);
    }

    for (int j=0; j<schd.Ncoords.size(); j++) {
      RiN(i, j) = pow( pow(d.coord[i][0] - schd.Ncoords[j][0], 2) +
                       pow(d.coord[i][1] - schd.Ncoords[j][1], 2) +
                       pow(d.coord[i][2] - schd.Ncoords[j][2], 2), 0.5);
    }

    updateGradAndLaplaceRatio(i, oldcoord, cps, d);
  }


  //the position of the ith electron has changed
  double OverlapRatio(int i, Vector3d& coord, const rJastrow& cps,
                      const rDeterminant &d) const
  {
    double diff = 0.0;
    for (int t=0; t<cps.Terms.size(); t++) {
      diff += cps.Terms[t]->exponentDiff(i, coord, d);
    }
    return exp(diff);
  }


  void OverlapWithGradient(const rDeterminant& d, 
                           const rJastrow& cps,
                           VectorXd& grad,
                           const double& ovlp) const {
    
    if (schd.optimizeCps) {
      int index = 0;
      for (int t=0; t<cps.Terms.size(); t++) 
        cps.Terms[t]->OverlapWithGradient(Rij, RiN, d, grad, ovlp, index); 

    }
  }

};  


#endif
