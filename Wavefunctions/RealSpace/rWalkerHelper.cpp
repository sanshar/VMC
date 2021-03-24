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


#include "rWalkerHelper.h"
#include "global.h"
#include "input.h"
#include "JastrowTermsHardCoded.h"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

rWalkerHelper<rSlater>::rWalkerHelper(const rSlater &w, const rDeterminant &d) 
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

void rWalkerHelper<rSlater>::initInvDetsTables(const rSlater& w, const rDeterminant &d) {
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);

  DetMatrix[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  DetMatrix[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Gradient[0] = MatrixXd::Zero(d.nelec, d.nelec);
  Gradient[1] = MatrixXd::Zero(d.nelec, d.nelec);
  Gradient[2] = MatrixXd::Zero(d.nelec, d.nelec);

  Laplacian = MatrixXd::Zero(d.nelec, d.nelec);

  AO = MatrixXd::Zero(d.nelec, norbs);
  AOLaplacian = MatrixXd::Zero(d.nelec, norbs);
  AOGradient[0]  = MatrixXd::Zero(d.nelec, norbs);
  AOGradient[1]  = MatrixXd::Zero(d.nelec, norbs);
  AOGradient[2]  = MatrixXd::Zero(d.nelec, norbs);

  for (int elec = 0; elec < d.nalpha; elec++)
  {
    schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

    for (int j=0; j<norbs; j++)
    {
      AO(elec, j) = aoValues[j];
      AOGradient[0](elec, j) = aoValues[1*norbs+j];
      AOGradient[1](elec, j) = aoValues[2*norbs+j];
      AOGradient[2](elec, j) = aoValues[3*norbs+j];
      AOLaplacian  (elec, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
    }

    for (int mo = 0; mo < d.nalpha; mo++) 
    {
      for (int j = 0; j < norbs; j++)
      {
        DetMatrix[0](elec, mo) += aoValues[j] * w.getHforbs(0)(j, mo);

        Laplacian(elec, mo) += (aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j]) * w.getHforbs(0)(j, mo);

        Gradient[0](elec, mo) += aoValues[1*norbs+j] * w.getHforbs(0)(j, mo);
        Gradient[1](elec, mo) += aoValues[2*norbs+j] * w.getHforbs(0)(j, mo);
        Gradient[2](elec, mo) += aoValues[3*norbs+j] * w.getHforbs(0)(j, mo);
      }          
    }
  }
    
  if (d.nalpha != 0) {
    Eigen::FullPivLU<MatrixXcd> lua(DetMatrix[0]);
    if (lua.isInvertible()) {
      thetaInv[0] = lua.inverse();
      thetaDet[0][0] = lua.determinant();
    }
    else {
      cout << " overlap with alpha determinant not invertible" << endl;
      exit(0);
    }
  }
  else {
    thetaDet[0][0] = 1.0;
  }

    
  for (int elec = 0; elec < d.nbeta; elec++)
  {
    schd.basis->eval_deriv2(d.coord[elec + d.nalpha], &aoValues[0]);
    for (int j=0; j<norbs; j++)
    {
      AO(elec + d.nalpha, j) = aoValues[j];
      AOGradient[0](elec + d.nalpha, j) = aoValues[1*norbs+j];
      AOGradient[1](elec + d.nalpha, j) = aoValues[2*norbs+j];
      AOGradient[2](elec + d.nalpha, j) = aoValues[3*norbs+j];
      AOLaplacian  (elec + d.nalpha, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
    }

    for (int mo = 0; mo < d.nbeta; mo++) 
    {
      for (int j = 0; j < norbs; j++)
      {
        DetMatrix[1](elec, mo) += aoValues[j] * w.getHforbs(1)(j, mo);

        Laplacian(elec + d.nalpha, mo + d.nalpha) += (aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j] ) * w.getHforbs(1)(j, mo);
      
        Gradient[0](elec + d.nalpha, mo + d.nalpha) += aoValues[1*norbs+j] * w.getHforbs(1)(j, mo);
        Gradient[1](elec + d.nalpha, mo + d.nalpha) += aoValues[2*norbs+j] * w.getHforbs(1)(j, mo);
        Gradient[2](elec + d.nalpha, mo + d.nalpha) += aoValues[3*norbs+j] * w.getHforbs(1)(j, mo);
      }
    }
  }


  if (d.nbeta != 0) {
    Eigen::FullPivLU<MatrixXcd> lub(DetMatrix[1]);
    if (lub.isInvertible()) {
      thetaInv[1] = lub.inverse();
      thetaDet[0][1] = lub.determinant();
    }
    else {
      cout << " overlap with beta determinant not invertible" << endl;
      exit(0);
    }
  }
  else {
      thetaDet[0][1] = 1.0;
  }
}

void rWalkerHelper<rSlater>::initInvDetsTablesGhf(const rSlater& w, const rDeterminant &d) {
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);

  DetMatrix[0] = MatrixXcd::Zero(d.nelec, d.nelec);
  Gradient [0] = MatrixXcd::Zero(d.nelec, d.nelec);
  Gradient [1] = MatrixXcd::Zero(d.nelec, d.nelec);
  Gradient [2] = MatrixXcd::Zero(d.nelec, d.nelec);
  Laplacian = MatrixXcd::Zero(d.nelec, d.nelec);

  AO = MatrixXd::Zero(d.nelec, 2*norbs);
  AOLaplacian = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[0]  = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[1]  = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[2]  = MatrixXd::Zero(d.nelec, 2*norbs);
    
  for (int elec=0; elec<d.nelec; elec++) {
    schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

    for (int j=0; j<norbs; j++) {
      int J = elec < d.nalpha ? j : j+norbs;
      AO(elec, J) = aoValues[j];
      AOGradient[0](elec, J) = aoValues[1*norbs+j];
      AOGradient[1](elec, J) = aoValues[2*norbs+j];
      AOGradient[2](elec, J) = aoValues[3*norbs+j];
      AOLaplacian  (elec, J) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
    }
    
    for (int mo=0; mo<d.nelec; mo++) 
      for (int j=0; j<norbs; j++) {
        int J = elec < d.nalpha ? j : j+norbs;
        DetMatrix[0](elec, mo) += aoValues[j] * (w.getHforbs(0)(J, mo));

        Laplacian(elec, mo) += (aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j]) * (w.getHforbs(0)(J,mo));

        Gradient[0](elec, mo) += aoValues[1*norbs+j] * (w.getHforbs(0)(J,mo));
        Gradient[1](elec, mo) += aoValues[2*norbs+j] * (w.getHforbs(0)(J,mo));
        Gradient[2](elec, mo) += aoValues[3*norbs+j] * (w.getHforbs(0)(J,mo)); 
      }
  }

  Eigen::FullPivLU<MatrixXcd> lu(DetMatrix[0]);
  if (lu.isInvertible()) {
    thetaInv[0] = lu.inverse();
    thetaDet[0][0] = lu.determinant();
    thetaDet[0][1] = 1.0;
  }
  else {
      if (commrank == 0) {
    cout << " overlap with GHF determinant not invertible" << endl;
    cout << w.getHforbs(0)<<endl;
    cout << DetMatrix[0].rows()<<"  "<<DetMatrix[0].cols()<<endl;
    cout << DetMatrix[0]<<endl<<endl;
    cout << thetaInv[0]<<endl<<endl;
    exit(0);
      }
  }
}

double rWalkerHelper<rSlater>::getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d, const rSlater& w) const
{
  if (hftype == Generalized) 
    return getDetFactorGHF(i, newCoord, 0, d.nelec, w);
  else if (i < d.nalpha)
    return getDetFactor(i, newCoord, 0, d.nalpha, w);
  else
    return getDetFactor(i-d.nalpha, newCoord, 1, d.nbeta, w);
}

double rWalkerHelper<rSlater>::getDetFactorGHF(int i, Vector3d& newCoord, int sz, int nelec, const rSlater& w) const
{
  int norbs = Determinant::norbs;
  aoValues.resize(norbs);

  schd.basis->eval(newCoord, &aoValues[0]);

  VectorXcd newVec = VectorXcd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) {
      int J = i < rDeterminant::nalpha ? j : j+norbs;
      newVec(mo) += aoValues[j] * (w.getHforbs(sz)(J, mo));
    }
  std::complex<double> factor = newVec.transpose() * thetaInv[sz].col(i);  
  return (factor * thetaDet[0][0]).real() / (thetaDet[0][0]).real();
}
  
double rWalkerHelper<rSlater>::getDetFactor(int i, Vector3d& newCoord, int sz, int nelec, const rSlater& w) const
{
  int norbs = Determinant::norbs;
  aoValues.resize(norbs);

  schd.basis->eval(newCoord, &aoValues[0]);

  VectorXcd newVec = VectorXcd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) 
      newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);

  std::complex<double> factor = newVec.transpose() * thetaInv[sz].col(i);  
  return (factor * thetaDet[0][0] * thetaDet[0][1]).real() / (thetaDet[0][0] * thetaDet[0][1]).real();
}

void rWalkerHelper<rSlater>::updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d,
                                         const rSlater& w) {
  if (hftype == Generalized) 
    updateWalkerGHF(i, oldCoord, d, 0, d.nelec, w);
  else if (i < d.nalpha)
    updateWalker(i, oldCoord, d, 0, d.nalpha, w);
  else
    updateWalker(i-d.nalpha, oldCoord, d, 1, d.nbeta, w);
}


void rWalkerHelper<rSlater>::updateWalkerGHF(int elec, Vector3d& oldCoord, const rDeterminant &d,
                                            int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(10 * norbs, 0.0);

  schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

  for (int j=0; j<norbs; j++) {
      int J = elec < d.nalpha ? j : j+norbs;
      AO(elec, J) = aoValues[j];
      AOGradient[0](elec, J) = aoValues[1*norbs+j];
      AOGradient[1](elec, J) = aoValues[2*norbs+j];
      AOGradient[2](elec, J) = aoValues[3*norbs+j];
      AOLaplacian  (elec, J) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
  }
  
  VectorXcd newVec = VectorXcd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) {
      int J = elec < rDeterminant::nalpha ? j : j+norbs;
      newVec(mo) += aoValues[j] * (w.getHforbs(sz)(J, mo));
    }
    
  calculateInverseDeterminantWithRowChange(thetaInv[sz],thetaDet[0][sz],DetMatrix[sz], elec, newVec);

  for (int mo=0; mo<nelec; mo++) {
    Laplacian(elec, mo) = 0.0;
    Gradient[0](elec, mo) = 0.0;
    Gradient[1](elec, mo) = 0.0;
    Gradient[2](elec, mo) = 0.0;
      
    for (int j=0; j<norbs; j++) {
      int J = elec < rDeterminant::nalpha ? j : j+norbs;
      Laplacian(elec, mo) += (aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j]) * (w.getHforbs(sz)(J, mo)); 
      Gradient[0](elec, mo) += aoValues[1*norbs+j] * (w.getHforbs(sz)(J, mo));
      Gradient[1](elec, mo) += aoValues[2*norbs+j] * (w.getHforbs(sz)(J, mo));
      Gradient[2](elec, mo) += aoValues[3*norbs+j] * (w.getHforbs(sz)(J, mo));
    } 
  }

}
  
  
void rWalkerHelper<rSlater>::updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d,
                                         int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(10 * norbs, 0.0);

  int gelec = elec;
  if (sz == 1) gelec += d.nalpha; //beta electron

  schd.basis->eval_deriv2(d.coord[gelec], &aoValues[0]);

  for (int j=0; j<norbs; j++) {
      AO(gelec, j) = aoValues[j];
      AOGradient[0](gelec, j) = aoValues[1*norbs+j];
      AOGradient[1](gelec, j) = aoValues[2*norbs+j];
      AOGradient[2](gelec, j) = aoValues[3*norbs+j];
      AOLaplacian  (gelec, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
  }

  VectorXcd newVec = VectorXd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++)  
      newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);
 
  calculateInverseDeterminantWithRowChange(thetaInv[sz], thetaDet[0][sz], DetMatrix[sz], elec, newVec);

  int shift = sz == 0 ? 0 : d.nalpha;
  for (int mo=0; mo<nelec; mo++) {
    Laplacian(gelec, mo + shift) = 0.0;
    Gradient[0](gelec, mo + shift) = 0.0;
    Gradient[1](gelec, mo + shift) = 0.0;
    Gradient[2](gelec, mo + shift) = 0.0;
      
    for (int j=0; j<norbs; j++) {
      Laplacian(gelec, mo + shift) += (aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j] ) * w.getHforbs(sz)(j,mo); 
      Gradient[0](gelec, mo + shift) += aoValues[1*norbs+j] * w.getHforbs(sz)(j, mo);
      Gradient[1](gelec, mo + shift) += aoValues[2*norbs+j] * w.getHforbs(sz)(j, mo);
      Gradient[2](gelec, mo + shift) += aoValues[3*norbs+j] * w.getHforbs(sz)(j, mo);
    } 
  }

}


void rWalkerHelper<rSlater>::OverlapWithGradient(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad,
                                                const double& ovlp) 
{
  if (schd.optimizeOrbs) {
    if (schd.hf == "ghf")
      OverlapWithGradientGhf(d, ref, grad);
    else
      OverlapWithGradient(d, ref, grad);
  }
}

void rWalkerHelper<rSlater>::HamOverlap(const rDeterminant& d, 
                                        const rSlater& ref,
                                        MatrixXd& Rij,
                                        MatrixXd& RiN,
                                        Eigen::VectorBlock<VectorXd>& hamgrad)
{


}


void rWalkerHelper<rSlater>::OverlapWithGradient(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad)
{
  grad.setZero();
  
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();
  
  std::complex<double> DetFactor = thetaDet[0][0] * thetaDet[0][1];
  //Assuming a single determinant
  for (int moa=0; moa<nalpha; moa++) {//alpha mo 
    for (int orb=0; orb<norbs; orb++) {//ao
      std::complex<double> factor = thetaInv[0].row(moa) * AO.col(orb).head(nalpha);
      factor *= DetFactor / DetFactor.real();
      grad[numDets + 2*orb * nalpha + 2*moa] += factor.real();
      grad[numDets + 2*orb * nalpha + 2*moa + 1] += -factor.imag();
    }
  }
  
  for (int mob=0; mob<nbeta; mob++) {//beta mo 
    for (int orb=0; orb<norbs; orb++) {//ao
        std::complex<double> factor = thetaInv[1].row(mob) * AO.col(orb).tail(nbeta);
        factor *= DetFactor / DetFactor.real();
      if (ref.hftype == Restricted) {
        grad[numDets + 2*orb * nbeta + 2*mob] += factor.real();
        grad[numDets + 2*orb * nbeta + 2*mob + 1] += -factor.imag();
      }
      else {
        grad[numDets + 2*nalpha*norbs + 2*orb * nbeta + 2*mob] += factor.real();
        grad[numDets + 2*nalpha*norbs + 2*orb * nbeta + 2*mob + 1] += -factor.imag();
      }
    }
  }
  

}


void rWalkerHelper<rSlater>::OverlapWithGradientGhf(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad) 
{
  grad.setZero();
  
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();
  std::complex<double> i(0.0, 1.0);
  
  for (int mo=0; mo<nelec; mo++) {
    for (int orb=0; orb< 2*norbs; orb++) {
      std:complex<double> factor = thetaInv[0].row(mo) * AO.col(orb);
      grad[numDets + 2*orb * nelec + 2*mo] = (factor * thetaDet[0][0]).real() / thetaDet[0][0].real();
      if (schd.ifComplex) grad[numDets + 2*orb * nelec + 2*mo + 1] = (i * factor * thetaDet[0][0]).real() / thetaDet[0][0].real();
    }
  }
}

//********************** MultiSlater ******************
rWalkerHelper<rMultiSlater>::rWalkerHelper(const rMultiSlater &w, const rDeterminant &d)
{
  initInvDetsTables(w, d);
}

void rWalkerHelper<rMultiSlater>::initInvDetsTables(const rMultiSlater &w, const rDeterminant &d)
{
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);

  A[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  A[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Ainv[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  Ainv[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Abar[0] = MatrixXd::Zero(d.nalpha, w.open[0].size());
  Abar[1] = MatrixXd::Zero(d.nbeta, w.open[1].size());

  Lap[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  Lap[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Lapbar[0] = MatrixXd::Zero(d.nalpha, w.open[0].size());
  Lapbar[1] = MatrixXd::Zero(d.nbeta, w.open[1].size());

  Grad[0][0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  Grad[0][1] = MatrixXd::Zero(d.nalpha, d.nalpha);
  Grad[0][2] = MatrixXd::Zero(d.nalpha, d.nalpha);
  Grad[1][0] = MatrixXd::Zero(d.nbeta, d.nbeta);
  Grad[1][1] = MatrixXd::Zero(d.nbeta, d.nbeta);
  Grad[1][2] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Gradbar[0][0] = MatrixXd::Zero(d.nalpha, w.open[0].size());
  Gradbar[0][1] = MatrixXd::Zero(d.nalpha, w.open[0].size());
  Gradbar[0][2] = MatrixXd::Zero(d.nalpha, w.open[0].size());
  Gradbar[1][0] = MatrixXd::Zero(d.nbeta, w.open[1].size());
  Gradbar[1][1] = MatrixXd::Zero(d.nbeta, w.open[1].size());
  Gradbar[1][2] = MatrixXd::Zero(d.nbeta, w.open[1].size());

  AO[0] = MatrixXd::Zero(d.nalpha, norbs);
  AO[1] = MatrixXd::Zero(d.nbeta, norbs);

  AOLap[0] = MatrixXd::Zero(d.nalpha, norbs);
  AOLap[1] = MatrixXd::Zero(d.nbeta, norbs);

  AOGrad[0][0] = MatrixXd::Zero(d.nalpha, norbs);
  AOGrad[0][1] = MatrixXd::Zero(d.nalpha, norbs);
  AOGrad[0][2] = MatrixXd::Zero(d.nalpha, norbs);

  AOGrad[1][0] = MatrixXd::Zero(d.nbeta, norbs);
  AOGrad[1][1] = MatrixXd::Zero(d.nbeta, norbs);
  AOGrad[1][2] = MatrixXd::Zero(d.nbeta, norbs);

  //alpha
  for (int elec = 0; elec < d.nalpha; elec++)
  {
    schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

    for (int j = 0; j < norbs; j++)
    {
      AO[0](elec, j) = aoValues[j];

      AOGrad[0][0](elec, j) = aoValues[1*norbs+j];
      AOGrad[0][1](elec, j) = aoValues[2*norbs+j];
      AOGrad[0][2](elec, j) = aoValues[3*norbs+j];

      AOLap[0](elec, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
    }
  }

  //occ
  for (int mo = 0; mo < w.ref[0].size(); mo++)
  {
    int orb = w.ref[0].at(mo);

    A[0].col(mo) = AO[0] * w.HforbsA.col(orb);

    Grad[0][0].col(mo) = AOGrad[0][0] * w.HforbsA.col(orb);
    Grad[0][1].col(mo) = AOGrad[0][1] * w.HforbsA.col(orb);
    Grad[0][2].col(mo) = AOGrad[0][2] * w.HforbsA.col(orb);

    Lap[0].col(mo) = AOLap[0] * w.HforbsA.col(orb);
  }

  //unocc
  for (int mo = 0; mo < w.open[0].size(); mo++)
  {
    int orb = w.open[0].at(mo);

    Abar[0].col(mo) = AO[0] * w.HforbsA.col(orb);

    Gradbar[0][0].col(mo) = AOGrad[0][0] * w.HforbsA.col(orb);
    Gradbar[0][1].col(mo) = AOGrad[0][1] * w.HforbsA.col(orb);
    Gradbar[0][2].col(mo) = AOGrad[0][2] * w.HforbsA.col(orb);

    Lapbar[0].col(mo) = AOLap[0] * w.HforbsA.col(orb);
  }

  if (d.nalpha != 0) {
    Eigen::FullPivLU<MatrixXcd> lua(A[0]);
    if (lua.isInvertible()) {
      Ainv[0] = lua.inverse();
      detA[0] = lua.determinant();
      AinvAbar[0] = Ainv[0] * Abar[0];
    }
    else {
      cout << " overlap with alpha determinant not invertible" << endl;
      exit(0);
    }
  }
  else { detA[0] = 1.0; }


  //beta
  for (int elec = 0; elec < d.nbeta; elec++)
  {
    schd.basis->eval_deriv2(d.coord[elec + d.nalpha], &aoValues[0]);

    for (int j=0; j<norbs; j++)
    {
      AO[1](elec, j) = aoValues[j];

      AOGrad[1][0](elec, j) = aoValues[1*norbs+j];
      AOGrad[1][1](elec, j) = aoValues[2*norbs+j];
      AOGrad[1][2](elec, j) = aoValues[3*norbs+j];

      AOLap[1](elec, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
    }
  }

  //occ
  for (int mo = 0; mo < w.ref[1].size(); mo++)
  {
    int orb = w.ref[1].at(mo);

    A[1].col(mo) = AO[1] * w.HforbsB.col(orb);

    Grad[1][0].col(mo) = AOGrad[1][0] * w.HforbsB.col(orb);
    Grad[1][1].col(mo) = AOGrad[1][1] * w.HforbsB.col(orb);
    Grad[1][2].col(mo) = AOGrad[1][2] * w.HforbsB.col(orb);

    Lap[1].col(mo) = AOLap[1] * w.HforbsB.col(orb);
  }

  //unocc
  for (int mo = 0; mo < w.open[1].size(); mo++)
  {
    int orb = w.open[1].at(mo);

    Abar[1].col(mo) = AO[1] * w.HforbsB.col(orb);

    Gradbar[1][0].col(mo) = AOGrad[1][0] * w.HforbsB.col(orb);
    Gradbar[1][1].col(mo) = AOGrad[1][1] * w.HforbsB.col(orb);
    Gradbar[1][2].col(mo) = AOGrad[1][2] * w.HforbsB.col(orb);

    Lapbar[1].col(mo) = AOLap[1] * w.HforbsB.col(orb);
  }
 
  if (d.nbeta != 0) {
    Eigen::FullPivLU<MatrixXcd> lub(A[1]);
    if (lub.isInvertible()) {
      Ainv[1] = lub.inverse();
      detA[1] = lub.determinant();
      AinvAbar[1] = Ainv[1] * Abar[1];
    }
    else {
      cout << " overlap with beta determinant not invertible" << endl;
      exit(0);
    }
  }
  else { detA[1] = 1.0; }


  //totalRatio and detRatios
  std::array<std::complex<double>, 2> ratio;
  ratio[0] = 1.0;
  ratio[1] = 1.0;
  detRatios.push_back(ratio);
  std::complex<double> d0 = detA[0] * detA[1];
  double c0 = w.ciParity[0] * w.ciCoeffs[0];

  std::complex<double> sum = 0.0;
  for (int I = 1; I < w.getNumOfDets(); I++)
  {
    //alpha
    Eigen::VectorXi rowVecA = w.ciIndices[0][I][0];
    Eigen::VectorXi colVecA = w.ciIndices[0][I][1];

    Eigen::MatrixXcd aA;
    igl::slice(AinvAbar[0], rowVecA, colVecA, aA);

    Eigen::FullPivLU<MatrixXcd> lua(aA);
    std::complex<double> ratioA = lua.determinant();

    //beta
    Eigen::VectorXi rowVecB = w.ciIndices[1][I][0];
    Eigen::VectorXi colVecB = w.ciIndices[1][I][1];

    Eigen::MatrixXcd aB;
    igl::slice(AinvAbar[1], rowVecB, colVecB, aB);

    Eigen::FullPivLU<MatrixXcd> lub(aB);
    std::complex<double> ratioB = lub.determinant();

    //alpha and beta
    double cI = w.ciParity[I] * w.ciCoeffs[I];
    std::complex<double> ratioI = ratioA * ratioB;

    sum += cI * ratioI;
    ratio[0] = ratioA;
    ratio[1] = ratioB;
    detRatios.push_back(ratio);
  }
  totalRatio = c0 + sum;
}

double rWalkerHelper<rMultiSlater>::getDetFactor(int elec, Vector3d& newCoord, const rDeterminant &d, const rMultiSlater& w) const
{
  int norbs = Determinant::norbs;
  aoValues.resize(norbs);

  int sz = 0;
  if (elec >= d.nalpha) //beta electron
  {
    elec -= d.nalpha;
    sz = 1;
  }

  schd.basis->eval(newCoord, &aoValues[0]);
  VectorXcd newAO = VectorXd::Zero(norbs);
  for (int i = 0; i < norbs; i++) { newAO(i) = aoValues[i]; }

  //occ
  VectorXcd newA = VectorXd::Zero(w.ref[sz].size());
  for (int mo = 0; mo < newA.size(); mo++)
  {
    int orb = w.ref[sz].at(mo);
    newA(mo) = newAO.transpose() * w.getHforbs(sz).col(orb);
  }

  //unocc
  VectorXcd newAbar = VectorXd::Zero(w.open[sz].size());
  for (int mo = 0; mo < newAbar.size(); mo++)
  {
    int orb = w.open[sz].at(mo);
    newAbar(mo) = newAO.transpose() * w.getHforbs(sz).col(orb);
  }

  //intermediates
  MatrixXcd Mbar = Ainv[sz].col(elec) * newAbar.transpose();
  VectorXcd temp;
  temp.transpose() = newA.transpose() * AinvAbar[sz];
  Mbar -= Ainv[sz].col(elec) * temp.transpose();

  std::complex<double> refRatio = newA.transpose() * Ainv[sz].col(elec);  

  std::complex<double> sum = 0.0;
  for (size_t I = 1; I < w.getNumOfDets(); I++)
  {
    Eigen::VectorXi rowVec = w.ciIndices[sz][I][0];
    Eigen::VectorXi colVec = w.ciIndices[sz][I][1];

    Eigen::MatrixXcd aI;
    igl::slice(AinvAbar[sz], rowVec, colVec, aI);

    Eigen::MatrixXcd MbarI;
    igl::slice(Mbar, rowVec, colVec, MbarI);

    //alpha and beta
    double cI = w.ciParity[I] * w.ciCoeffs[I];
    std::complex<double> ratioI = detRatios[I][0] * detRatios[I][1];

    Eigen::FullPivLU<MatrixXcd> lu(aI);
    if (!lu.isInvertible()) { continue; }
    Eigen::MatrixXcd aIinv = lu.inverse();

    sum += cI * (aIinv * MbarI).trace() * ratioI;
  }

  return (refRatio + sum / totalRatio).real();
}

void rWalkerHelper<rMultiSlater>::updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d, const rMultiSlater& w)
{
  int norbs = Determinant::norbs;
  aoValues.resize(10 * norbs, 0.0);
  schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

  int sz = 0;
  if (elec >= d.nalpha) //beta electron
  {
    elec -= d.nalpha;
    sz = 1;
  }

  for (int j = 0; j < norbs; j++)
  {
    AO[sz](elec, j) = aoValues[j];

    AOGrad[sz][0](elec, j) = aoValues[1*norbs+j];
    AOGrad[sz][1](elec, j) = aoValues[2*norbs+j];
    AOGrad[sz][2](elec, j) = aoValues[3*norbs+j];

    AOLap[sz](elec, j) = aoValues[4*norbs+j] + aoValues[7*norbs+j] + aoValues[9*norbs+j];
  }

  //occ
  Eigen::VectorXcd oldArow = A[sz].row(elec).transpose();
  for (int mo = 0; mo < w.ref[sz].size(); mo++)
  {
    int orb = w.ref[sz].at(mo);

    A[sz](elec, mo) = AO[sz].row(elec) * w.getHforbs(sz).col(orb);

    Grad[sz][0](elec, mo) = AOGrad[sz][0].row(elec) * w.getHforbs(sz).col(orb);
    Grad[sz][1](elec, mo) = AOGrad[sz][1].row(elec) * w.getHforbs(sz).col(orb);
    Grad[sz][2](elec, mo) = AOGrad[sz][2].row(elec) * w.getHforbs(sz).col(orb);

    Lap[sz](elec, mo) = AOLap[sz].row(elec) * w.getHforbs(sz).col(orb);
  }
  Eigen::VectorXcd newArow = A[sz].row(elec).transpose();

  //unocc
  Eigen::VectorXcd oldAbarrow = Abar[sz].row(elec).transpose();
  for (int mo = 0; mo < w.open[sz].size(); mo++)
  {
    int orb = w.open[sz].at(mo);

    Abar[sz](elec, mo) = AO[sz].row(elec) * w.getHforbs(sz).col(orb);

    Gradbar[sz][0](elec, mo) = AOGrad[sz][0].row(elec) * w.getHforbs(sz).col(orb);
    Gradbar[sz][1](elec, mo) = AOGrad[sz][1].row(elec) * w.getHforbs(sz).col(orb);
    Gradbar[sz][2](elec, mo) = AOGrad[sz][2].row(elec) * w.getHforbs(sz).col(orb);

    Lapbar[sz](elec, mo) = AOLap[sz].row(elec) * w.getHforbs(sz).col(orb);
  }
  Eigen::VectorXcd newAbarrow = Abar[sz].row(elec).transpose();

  //new Ainv
  Eigen::MatrixXcd newAinv = Ainv[sz];
  Eigen::VectorXcd deltaA = newArow - oldArow;
  Eigen::VectorXcd temp;
  temp.transpose() = deltaA.transpose() * Ainv[sz];
  std::complex<double> denom = 1.0 + deltaA.transpose() * Ainv[sz].col(elec);
  newAinv -= Ainv[sz].col(elec) * temp.transpose() / denom;

  //new AinvAbar
  //1
  Eigen::MatrixXcd newAinvAbar = AinvAbar[sz];
  //2
  Eigen::VectorXcd deltaAbar = newAbarrow - oldAbarrow;
  newAinvAbar += Ainv[sz].col(elec) * deltaAbar.transpose();
  //3
  temp.transpose() = deltaA.transpose() * AinvAbar[sz];
  newAinvAbar -= Ainv[sz].col(elec) * temp.transpose() / denom;
  //4
  std::complex<double> middle = deltaA.transpose() * Ainv[sz].col(elec);
  newAinvAbar -= middle * Ainv[sz].col(elec) * deltaAbar.transpose() / denom;

  //update walker variables
  Ainv[sz] = newAinv;
  AinvAbar[sz] = newAinvAbar;
  detA[sz] = detA[sz] * denom;

  //recalculate totalRatio and detRatios
  std::array<std::complex<double>, 2> ratio;
  ratio[0] = 1.0;
  ratio[1] = 1.0;
  detRatios.at(0) = ratio;
  std::complex<double> d0 = detA[0] * detA[1];
  double c0 = w.ciParity[0] * w.ciCoeffs[0];

  std::complex<double> sum = 0.0;
  for (int I = 1; I < w.getNumOfDets(); I++)
  {
    //alpha
    Eigen::VectorXi rowVecA = w.ciIndices[0][I][0];
    Eigen::VectorXi colVecA = w.ciIndices[0][I][1];

    Eigen::MatrixXcd aA;
    igl::slice(AinvAbar[0], rowVecA, colVecA, aA);
    std::complex<double> ratioA = aA.determinant();

    //beta
    Eigen::VectorXi rowVecB = w.ciIndices[1][I][0];
    Eigen::VectorXi colVecB = w.ciIndices[1][I][1];

    Eigen::MatrixXcd aB;
    igl::slice(AinvAbar[1], rowVecB, colVecB, aB);
    std::complex<double> ratioB = aB.determinant();

    //alpha and beta
    double cI = w.ciParity[I] * w.ciCoeffs[I];
    std::complex<double> ratioI = ratioA * ratioB;

    sum += cI * ratioI;
    ratio[0] = ratioA;
    ratio[1] = ratioB;
    detRatios.at(I) = ratio;
  }
  totalRatio = c0 + sum;
}

void rWalkerHelper<rMultiSlater>::OverlapWithGradient(const rDeterminant& d, const rMultiSlater& w, Eigen::VectorBlock<VectorXd>& grad) const
{
  int norbs = schd.basis->getNorbs();
  int nact = norbs;
  if (schd.nciAct > 0) { nact = schd.nciAct; }
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha + nbeta;

  Eigen::VectorXd CIgradRatio = VectorXd::Zero(0);
  if (schd.optimizeCiCoeffs)
  {
    CIgradRatio.setZero(w.getNumOfDets());

    for (int I = 0; I < w.getNumOfDets(); I++) { CIgradRatio(I) = w.ciParity[I] * (detRatios[I][0] * detRatios[I][1] / totalRatio).real(); }
  }//ci param grad

  Eigen::VectorXd OrbgradRatio = VectorXd::Zero(0);
  if (schd.optimizeOrbs)
  {
    int num = nact * norbs; //num of orbital params
    OrbgradRatio.setZero(num);

    //build Gamma (J. Chem. Theory Comput. 2017, 13, 5273−5281)
    std::array<Eigen::MatrixXd, 2> R;
    R[0] = Eigen::MatrixXd::Zero(nact, nalpha);
    R[1] = Eigen::MatrixXd::Zero(nact, nbeta);
    std::array<Eigen::MatrixXcd, 2> Aall;
    Aall[0] = Eigen::MatrixXd::Zero(nalpha, nact);
    Aall[1] = Eigen::MatrixXd::Zero(nbeta, nact);
    for (int sz = 0; sz < 2; sz++)
    {
      for (int i = 0; i < w.ref[sz].size(); i++)
      {
        R[sz](w.ref[sz][i], i) = 1.0;

        Aall[sz].col(w.ref[sz][i]) = A[sz].col(i);
      }

      for (int i = 0; i < w.open[sz].size(); i++)
      {
        Aall[sz].col(w.open[sz][i]) = Abar[sz].col(i);
      }
    }
    std::array<Eigen::MatrixXcd, 2> AinvAall;
    AinvAall[0] = Ainv[0] * Aall[0];
    AinvAall[1] = Ainv[1] * Aall[1];
    std::array<Eigen::MatrixXcd, 2> alpha = AinvAbar;

    std::array<Eigen::MatrixXcd, 2> Y;
    Y[0] = Eigen::MatrixXd::Zero(nact, nalpha);
    Y[1] = Eigen::MatrixXd::Zero(nact, nbeta);
    for (int I = 1; I < w.getNumOfDets(); I++)
    {
      double cI = w.ciParity[I] * w.ciCoeffs[I];
      std::complex<double> ratioI = detRatios[I][0] * detRatios[I][1];

      for (int sz = 0; sz < 2; sz++)
      {
        //these are the orbital indices
        const Eigen::VectorXi &des = w.ciExcitations[sz][I][0];
        const Eigen::VectorXi &cre = w.ciExcitations[sz][I][1];

        //these are the internal indices
        Eigen::VectorXi rowVec = w.ciIndices[sz][I][0];
        Eigen::VectorXi colVec = w.ciIndices[sz][I][1];

        Eigen::MatrixXcd aI;
        igl::slice(alpha[sz], rowVec, colVec, aI);

        Eigen::FullPivLU<MatrixXcd> lu(aI);
        if (!lu.isInvertible()) { continue; }
        Eigen::MatrixXcd aIinv = lu.inverse();

        for (int i = 0; i < des.size(); i++)
        {
          for (int j = 0; j < cre.size(); j++)
          {
            Y[sz](cre(j), rowVec(i)) += cI * ratioI * aIinv(j, i);
          }
        }
      }//sz
    }//dets
    Y[0] /= totalRatio;
    Y[1] /= totalRatio;

    std::array<Eigen::MatrixXcd, 2> G;
    for (int sz = 0; sz < 2; sz++)
    {
      Eigen::MatrixXcd one = R[sz] * Ainv[sz];
      Eigen::MatrixXcd two = Y[sz] * Ainv[sz];
      Eigen::MatrixXcd three = R[sz] * AinvAall[sz];
      G[sz] = one + two - (three * two);
    }

    for (int p = 0; p < norbs; p++)
    {
      for (int q = 0; q < nact; q++)
      {
        for (int sz = 0; sz < 2; sz++)
        {
          //gradRatio
          std::complex<double> factor = G[sz].row(q) * AO[sz].col(p);
          OrbgradRatio(p * nact + q) += factor.real();
        }
      }
    }

  }//orb param grad

  grad << CIgradRatio, OrbgradRatio;
}

//********************** Backflow Slater ***************
rWalkerHelper<rBFSlater>::rWalkerHelper(const rBFSlater &w, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN) 
{
  //calculate dp, etaValues, gValues, and their derivatives
  //these will be updated with every mc move, and not calculated from scratch
  initPositionTables(w, d, Rij, RiN);
  
  //used directly in overlap ratio calculations 
  proposedDetMatrix = MatrixXcd::Zero(d.nelec, d.nelec);
  calcDetMatrix(w, dp);
  DetMatrix = proposedDetMatrix;
  thetaDet = DetMatrix.determinant();
  proposedThetaDet = thetaDet;
  
  //the mo derivatives and inverse tables will be calculated from scratch before each local energy calculation and will not be updated
  thetaInv = MatrixXcd::Zero(d.nelec, d.nelec);
  slaterLaplacianRatio = VectorXcd::Zero(d.nelec);
  for (int i = 0; i < 3; i++) {
    MOGradient[i] = MatrixXcd::Zero(d.nelec, d.nelec);
    slaterGradientRatio[i] = VectorXcd::Zero(d.nelec);
    rTable[i] = MatrixXcd::Zero(d.nelec, d.nelec);
  }
  for (int i = 0; i < 6; i++) {
    MOSecondDerivatives[i] = MatrixXcd::Zero(d.nelec, d.nelec);
  }
}

double calcHTerms(int i, const rBFSlater &w, const rDeterminant &d, const VectorXd &RiN, Array3d &Dh, Array3d &D2h) 
{
  double hi = 1.;
  vector<Array3d> DhVec;  //useful intermediate
  DhVec.resize(schd.Ncoords.size());
  for (int N = 0; N < schd.Ncoords.size(); N++) {
    double giN = w.gFun(RiN(N));
    hi *= giN;
    double DgDr = 10 * RiN(N) * (1 - giN);
    double D2gDr2 = 10 * (1 - 10 * pow(RiN(N), 2)) *  (1 - giN);
    Array3d Dr = (d.coord[i] - schd.Ncoords[N]).array() / RiN(N);
    Array3d D2r = 1 / RiN(N) - (d.coord[i] - schd.Ncoords[N]).array().square() / pow(RiN(N), 3);
    DhVec[N] = DgDr * Dr / giN;
    Dh += DhVec[N];
    D2h += (DgDr * D2r + D2gDr2 * Dr.square()) / giN;
    for (int M = 0; M < N; M++) {
      D2h += 2 * DhVec[M] * DhVec[N];
    }
  }
  Dh *= hi;
  D2h *= hi;
  return hi;
}

//Deta and D2eta are w.r.t. r_j components
double calcEtaTerms(int i, int j, const rBFSlater &w, const rDeterminant &d, const double &rij, Array3d &Deta, Array3d &D2eta) 
{
  bool sameSpinQ = (i + 0.5 - d.nalpha) * (j + 0.5 - d.nalpha) < 0 ? 0 : 1;
  double etaij = w.eta(rij, sameSpinQ);
  double bij = w.b[sameSpinQ]; 
  double DetaDr = -2 * rij * etaij / pow(bij, 2);
  double D2etaDr2 = 2 * etaij * (2 * pow(rij / bij, 2) - 1) / pow(bij, 2);
  Array3d Dr = (d.coord[j] - d.coord[i]).array() / rij;
  Array3d D2r = 1 / rij - (d.coord[j] - d.coord[i]).array().square() / pow(rij, 3);
  Deta = DetaDr * Dr;
  D2eta = DetaDr * D2r + D2etaDr2 * Dr.square();
  return etaij;
}

//Deta and D2eta are w.r.t. r_j components
double calcChiTerms(int i, int I, const rBFSlater &w, const rDeterminant &d, const double &riI, Array3d &Dchi, Array3d &D2chi) 
{
  double chiiI = w.chi(riI);
  double DchiDr = -2 * riI * chiiI / pow(w.bN, 2);
  double D2chiDr2 = 2 * chiiI * (2 * pow(riI / w.bN, 2) - 1) / pow(w.bN, 2);
  Array3d Dr = (d.coord[i] - schd.Ncoords[I]).array() / riI;
  Array3d D2r = 1 / riI - (d.coord[i] - schd.Ncoords[I]).array().square() / pow(riI, 3);
  Dchi = DchiDr * Dr;
  D2chi = DchiDr * D2r + D2chiDr2 * Dr.square();
  return chiiI;
}

void rWalkerHelper<rBFSlater>::initPositionTables(const rBFSlater &w, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN)
{
  dp = d;
  displacements.resize(d.nelec);
  gradaDisplacements.resize(d.nelec);
  gradbDisplacements.resize(d.nelec);
  gradaNDisplacements.resize(d.nelec);
  gradbNDisplacements.resize(d.nelec);
  for (int i = 0; i < d.nelec; i++) {
    displacements[i].setZero();
    gradaDisplacements[i][0].setZero();
    gradaDisplacements[i][1].setZero();
    gradbDisplacements[i][0].setZero();
    gradbDisplacements[i][1].setZero();
    gradaNDisplacements[i].setZero();
    gradbNDisplacements[i].setZero();
  }
  etaValues = MatrixXd::Zero(d.nelec, d.nelec);
  hValues = MatrixXd::Zero(d.nelec, schd.Ncoords.size());
  chiValues = MatrixXd::Zero(d.nelec, schd.Ncoords.size());
  for (int mu = 0; mu < 3; mu++) {
    hGradient[mu] = VectorXd::Zero(d.nelec);
    hSecondDerivatives[mu] = VectorXd::Zero(d.nelec);
  }
  for (int munu = 0; munu < 9; munu++) {
    rpGradient[munu] = MatrixXd::Zero(d.nelec, d.nelec);
    rpSecondDerivatives[munu] = MatrixXd::Zero(d.nelec, d.nelec);
  }//order: xx, xy, xz, yx, yy, yz, zx, zy, zz

  for (int i = 0; i < d.nelec; i++) {
    //hi term
    Array3d Dh(0., 0., 0.); 
    Array3d D2h(0., 0., 0.);
    hValues(i) = calcHTerms(i, w, d, RiN.row(i), Dh, D2h);
    for (int mu = 0; mu < 3; mu++) {
      hGradient[mu][i] = Dh(mu);
      hSecondDerivatives[mu][i] = D2h(mu);
    }
  
    //chi_iI terms
    for (int I = 0; I < schd.Ncoords.size(); I++) {
      Array3d Dchi(0., 0., 0.); 
      Array3d D2chi(0., 0., 0.);
      chiValues(i, I) = calcChiTerms(i, I, w, d, RiN(i, I), Dchi, D2chi);
      double chiiI = chiValues(i, I);
      //sans hi factors
      displacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]);
      gradaNDisplacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]) / w.aN;
      gradbNDisplacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]) * 2 * pow(RiN(i, I), 2) / pow(w.bN, 3);
      MatrixXd rDchi = (d.coord[i] - schd.Ncoords[I]) * Dchi.matrix().transpose();
      MatrixXd rD2chi = (d.coord[i] - schd.Ncoords[I]) * D2chi.matrix().transpose();
      for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
          int munu = 3 * mu + nu;
          double delta = mu == nu ? 1. : 0.;
          rpGradient[munu](i, i) += chiiI * delta + rDchi(mu, nu);
          rpSecondDerivatives[munu](i, i) += 2 * Dchi[nu] * delta + rD2chi(mu, nu);
        }
      }
    }

    //eta_ij terms
    //j < i contributions are calculated in previous loop iterations
    for (int j = i + 1; j < d.nelec; j++) {
      bool sameSpinQ = (i + 0.5 - d.nalpha) * (j + 0.5 - d.nalpha) < 0 ? 0 : 1;
      double aij = w.a[sameSpinQ], bij = w.b[sameSpinQ];
      Array3d Deta(0., 0., 0.); 
      Array3d D2eta(0., 0., 0.);
      etaValues(i, j) = calcEtaTerms(i, j, w, d, Rij(i, j), Deta, D2eta);
      etaValues(j, i) = etaValues(i, j);
      double etaij = etaValues(i, j);
 
      //sans hi factors, because these also serve as intermediates in second derivatrive calcs
      displacements[i] += etaij * (d.coord[i] - d.coord[j]);
      displacements[j] += etaij * (d.coord[j] - d.coord[i]);
      gradaDisplacements[i][sameSpinQ] += etaij * (d.coord[i] - d.coord[j]) / aij;
      gradaDisplacements[j][sameSpinQ] += etaij * (d.coord[j] - d.coord[i]) / aij;
      gradbDisplacements[i][sameSpinQ] += etaij * (d.coord[i] - d.coord[j]) * 2 * pow(Rij(i, j), 2) / pow(bij, 3);
      gradbDisplacements[j][sameSpinQ] += etaij * (d.coord[j] - d.coord[i]) * 2 * pow(Rij(i, j), 2) / pow(bij, 3);

      MatrixXd rDeta = (d.coord[i] - d.coord[j]) * Deta.matrix().transpose();
      MatrixXd rD2eta = (d.coord[i] - d.coord[j]) * D2eta.matrix().transpose();

      for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
          int munu = 3 * mu + nu;
          double delta = mu == nu ? 1. : 0.;

          rpGradient[munu](i, j) = -etaij * delta + rDeta(mu, nu);
          rpGradient[munu](j, i) = rpGradient[munu](i, j);
          rpGradient[munu](i, i) -= rpGradient[munu](i, j);
          rpGradient[munu](j, j) -= rpGradient[munu](i, j);
          
          rpSecondDerivatives[munu](i, j) = -2 * Deta[nu] * delta + rD2eta(mu, nu);
          rpSecondDerivatives[munu](j, i) = -rpSecondDerivatives[munu](i, j);
          rpSecondDerivatives[munu](i, i) += rpSecondDerivatives[munu](i, j);
          rpSecondDerivatives[munu](j, j) -= rpSecondDerivatives[munu](i, j);
        }
      }
    }

    //finish diagonal elements and include hi factor
    for (int mu = 0; mu < 3; mu++) {
      for (int nu = 0; nu < 3; nu++) {
        int munu = 3 * mu + nu;
        double delta = mu == nu ? 1. : 0.;
        rpSecondDerivatives[munu].row(i) *= hValues(i);
        rpSecondDerivatives[munu](i, i) += 2 * Dh(nu) * rpGradient[munu](i, i);
        rpSecondDerivatives[munu](i, i) += D2h(nu) * displacements[i](mu);
        rpGradient[munu].row(i) *= hValues(i);
        rpGradient[munu](i, i) += delta + Dh(nu) * displacements[i](mu);
      }
    }
    
    //finally add the backflow displacement
    dp.coord[i] += hValues(i) * displacements[i];
  }
}

void rWalkerHelper<rBFSlater>::calcDetMatrix(const rBFSlater& w, const rDeterminant &d) 
{
  proposedDetMatrix.setZero();
  int norbs = Determinant::norbs;
  aoValues.resize(norbs, 0.0);

  for (int elec = 0; elec < d.nelec; elec++) {
    schd.basis->eval(d.coord[elec], &aoValues[0]);
    for (int mo = 0; mo < d.nelec; mo++) {
      for (int j = 0; j < norbs; j++) {
        int J = elec < d.nalpha ? j : j+norbs;
        proposedDetMatrix(elec, mo) += aoValues[j] * (w.getHforbs(0)(J, mo));
      }
    }
  }
}

//to be used before a local energy calculation
void rWalkerHelper<rBFSlater>::calcSlaterDerivatives(const rBFSlater& w, const rDeterminant &d) 
{
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);
  for (int i = 0; i < 3; i++) MOGradient[i].setZero();
  for (int i = 0; i < 6; i++) MOSecondDerivatives[i].setZero();
  
  for (int elec = 0; elec < dp.nelec; elec++) {
    schd.basis->eval_deriv2(dp.coord[elec], &aoValues[0]);
    for (int mo = 0; mo < dp.nelec; mo++) { 
      for (int j = 0; j < norbs; j++) {
        int J = elec < dp.nalpha ? j : j+norbs;
        //DetMatrix(elec, mo) += aoValues[j] * (w.getHforbs(0)(J, mo));
        for (int i = 1; i < 4; i++) {
          MOGradient[i-1](elec, mo) += aoValues[i*norbs+j] * (w.getHforbs(0)(J,mo));
        }
        for (int i = 4; i < 10; i++) {
          MOSecondDerivatives[i-4](elec, mo) += aoValues[i*norbs+j] * (w.getHforbs(0)(J,mo));
        }
      }
    }
  }

  Eigen::FullPivLU<MatrixXcd> lu(DetMatrix);
  if (lu.isInvertible()) {
    thetaInv = lu.inverse();
    for (int i = 0; i < 3; i++) {
      rTable[i].noalias() = MOGradient[i] * thetaInv;
    }
  }
  else {
    if (commrank == 0) {
      cout << " overlap with GHF determinant not invertible" << endl;
      cout << w.getHforbs(0)<<endl;
      cout << "DetMatrix\n" << DetMatrix<<endl<<endl;
      cout << "proposedDetMatrix\n" << proposedDetMatrix<<endl<<endl;
      cout << thetaInv<<endl<<endl;
      cout << "d\n" << d << endl;
      cout << "thetaDet " << thetaDet << endl;
      exit(0);
    }
  }


  //calculate derivatives wrt bare coordinates
  //cout << "calcSlaterDerivatives\n";
  slaterGradientRatio[0].setZero();
  slaterGradientRatio[1].setZero();
  slaterGradientRatio[2].setZero();
  slaterLaplacianRatio.setZero();
  for (int i = 0; i < dp.nelec; i++) {
    for (int mu = 0; mu < 3; mu++) {
      for (int j = 0; j < dp.nelec; j++) {
        for (int nu = 0; nu < 3; nu++) {
          int numu = 3 * nu + mu;
          //cout << "i mu j nu " << i << " " << mu << " " << j << " " << nu << endl;
          slaterGradientRatio[mu][i] += rTable[nu](j, j) * rpGradient[numu](j, i);
          slaterLaplacianRatio[i] +=  rTable[nu](j, j) * rpSecondDerivatives[numu](j, i);
          for (int k = 0; k < dp.nelec; k++) {
            for (int lambda = 0; lambda < 3; lambda++) {
              int lambdamu = 3 * lambda + mu;
              //cout << "k lambda " << k << " " << lambda << endl;
              if (k == j) {//single row derivative
                int lambdanu = (min(lambda, nu) * (5 - min(lambda, nu))) / 2 + max(lambda, nu); //different kind of combined index
                //cout << "lambdanu " << lambdanu << endl;
                std::complex<double> detRatio =  MOSecondDerivatives[lambdanu].row(k) * thetaInv.col(k);
                //cout << "detRatio " << detRatio << endl;
                slaterLaplacianRatio[i] += detRatio * rpGradient[numu](j, i) * rpGradient[lambdamu](k, i);
              }
              else {//two rows
                int K = min(k, j); int J = max(k, j);
                int Lambda = lambda, Nu = nu;
                if (K != k) {
                  Lambda = nu;
                  Nu = lambda;
                }
                std::complex<double> detRatio = rTable[Lambda](K, K) * rTable[Nu](J, J) - rTable[Lambda](K, J) * rTable[Nu](J, K);
                //cout << "detRatio " << detRatio << endl;
                slaterLaplacianRatio[i] += detRatio * rpGradient[numu](j, i) * rpGradient[lambdamu](k, i);
              }
            }
          }
        }
      }
    }
  }

}

double rWalkerHelper<rBFSlater>::getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d, const rBFSlater& w) 
{
  //calculate h for the proposed position
  double hi = 1.;
  for(int N = 0; N < schd.Ncoords.size(); N++) {
    double giN = w.gFun((newCoord - schd.Ncoords[N]).norm());
    hi *= giN;
  }

  proposedDp = dp;
  proposedDp.coord[i] = newCoord;
  //calculate new backflow coordinates
  for (int I = 0; I < schd.Ncoords.size(); I++) {
    Vector3d riI = newCoord - schd.Ncoords[I];
    double chiiI = w.chi(riI.norm());
    proposedDp.coord[i] += hi * chiiI * riI;
  }
  
  for(int j = 0; j < d.nelec; j++) {
    if (i == j) continue;
    Vector3d rij = newCoord - d.coord[j];
    bool sameSpinQ = (i + 0.5 - d.nalpha) * (j + 0.5 - d.nalpha) < 0 ? 0 : 1;
    double etaij = w.eta(rij.norm(), sameSpinQ);
    proposedDp.coord[j] += hValues(j) * (etaij * (-rij) - etaValues(j, i) * (d.coord[j] - d.coord[i]));
    proposedDp.coord[i] += hi * etaij * rij;
  }

  calcDetMatrix(w, proposedDp);
  proposedThetaDet = proposedDetMatrix.determinant();
  return proposedThetaDet.real() / thetaDet.real();
}

//all arguments are updated
void rWalkerHelper<rBFSlater>::updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d, const MatrixXd &Rij, const MatrixXd &RiN,
                                         const rBFSlater& w) 
{
  thetaDet = proposedThetaDet;
  DetMatrix = proposedDetMatrix;
  dp = proposedDp;
  thetaDet = proposedThetaDet;

  //update position derivatives
  displacements[i].setZero();
  gradaDisplacements[i][0].setZero();
  gradaDisplacements[i][1].setZero();
  gradbDisplacements[i][0].setZero();
  gradbDisplacements[i][1].setZero();
  gradaNDisplacements[i].setZero();
  gradbNDisplacements[i].setZero();
  for (int mu = 0; mu < 3; mu++) {
    for (int nu = 0; nu < 3; nu++) {
      int munu = 3 * mu + nu;
      rpGradient[munu](i, i) = 0.;
      rpSecondDerivatives[munu](i, i) = 0.;
    }
  }
  Array3d Dh(0., 0., 0.); 
  Array3d D2h(0., 0., 0.);
  double hi = calcHTerms(i, w, d, RiN.row(i), Dh, D2h);
  hValues(i) = hi;

  //e-n terms
  for (int I = 0; I < schd.Ncoords.size(); I++) {
    Array3d Dchi(0., 0., 0.); 
    Array3d D2chi(0., 0., 0.);
    chiValues(i, I) = calcChiTerms(i, I, w, d, RiN(i, I), Dchi, D2chi);
    double chiiI = chiValues(i, I);
    //sans hi factors
    displacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]);
    gradaNDisplacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]) / w.aN;
    gradbNDisplacements[i] += chiiI * (d.coord[i] - schd.Ncoords[I]) * 2 * pow(RiN(i, I), 2) / pow(w.bN, 3);
    MatrixXd rDchi = (d.coord[i] - schd.Ncoords[I]) * Dchi.matrix().transpose();
    MatrixXd rD2chi = (d.coord[i] - schd.Ncoords[I]) * D2chi.matrix().transpose();
    for (int mu = 0; mu < 3; mu++) {
      for (int nu = 0; nu < 3; nu++) {
        int munu = 3 * mu + nu;
        double delta = mu == nu ? 1. : 0.;
        rpGradient[munu](i, i) += hi * (chiiI * delta + rDchi(mu, nu));
        rpSecondDerivatives[munu](i, i) += hi * (2 * Dchi[nu] * delta + rD2chi(mu, nu));
      }
    }
  }
  
  //e-e terms
  for(int j = 0; j < d.nelec; j++) {
    if (i == j) continue;
    Array3d Deta(0., 0., 0.); 
    Array3d D2eta(0., 0., 0.);
    double etaij = calcEtaTerms(i, j, w, d, Rij(i, j), Deta, D2eta);

    bool sameSpinQ = (i + 0.5 - d.nalpha) * (j + 0.5 - d.nalpha) < 0 ? 0 : 1;
    double aij = w.a[sameSpinQ], bij = w.b[sameSpinQ];
    displacements[i] += etaij * (d.coord[i] - d.coord[j]);
    gradaDisplacements[i][sameSpinQ] += etaij * (d.coord[i] - d.coord[j]) / aij;
    gradbDisplacements[i][sameSpinQ] += etaij * (d.coord[i] - d.coord[j]) * 2 * pow(Rij(i, j), 2) / pow(bij, 3);
    Vector3d deltaDj = etaij * (d.coord[j] - d.coord[i]) - etaValues(i, j) * (d.coord[j] - oldCoord);
    displacements[j] += deltaDj;
    gradaDisplacements[j][sameSpinQ] += deltaDj / aij;
    double oldRij = (d.coord[j] - oldCoord).norm();
    gradbDisplacements[j][sameSpinQ] += (etaij * (d.coord[j] - d.coord[i]) * pow(Rij(i, j), 2) - etaValues(i, j) * (d.coord[j] - oldCoord) * pow(oldRij, 2)) * 2 / pow(bij, 3);
    MatrixXd rDeta = (d.coord[i] - d.coord[j]) * Deta.matrix().transpose();
    MatrixXd rD2eta = (d.coord[i] - d.coord[j]) * D2eta.matrix().transpose();

    for (int mu = 0; mu < 3; mu++) {
      for (int nu = 0; nu < 3; nu++) {
        int munu = 3 * mu + nu;
        double delta = mu == nu ? 1. : 0.;

        //delta dd_{j\mu} / dr_{j\nu}
        double deltaDdj = (etaij * delta - rDeta(mu, nu)) + rpGradient[munu](j, i) / hValues(j);
        //delta d^2d_{j\mu} / dr_{j\nu}^2
        double deltaD2dj = (2 * Deta[nu] * delta - rD2eta(mu, nu)) - rpSecondDerivatives[munu](j, i) / hValues(j);

        //including h factors
        rpGradient[munu](j, j) += hGradient[nu][j] * deltaDj[mu] + hValues(j) * deltaDdj;
        rpGradient[munu](i, j) = hi * (-etaij * delta + rDeta(mu, nu));
        rpGradient[munu](j, i) = hValues(j) * rpGradient[munu](i, j) / hi;
        rpGradient[munu](i, i) -= rpGradient[munu](i, j);
        
        rpSecondDerivatives[munu](j, j) += hSecondDerivatives[nu][j] * deltaDj[mu] + hValues(j) * deltaD2dj + 2 * hGradient[nu][j] * deltaDdj;
        rpSecondDerivatives[munu](i, j) = (-2 * Deta[nu] * delta + rD2eta(mu, nu)) * hi;
        rpSecondDerivatives[munu](j, i) = -hValues(j) * rpSecondDerivatives[munu](i, j) / hi;
        rpSecondDerivatives[munu](i, i) += rpSecondDerivatives[munu](i, j);
      }
    }
    etaValues(i, j) = etaij;
    etaValues(j, i) = etaij;
  }

  //finish diagonal elements
  for (int mu = 0; mu < 3; mu++) {
    hGradient[mu][i] = Dh(mu);
    hSecondDerivatives[mu][i] = D2h(mu);
    for (int nu = 0; nu < 3; nu++) {
      int munu = 3 * mu + nu;
      double delta = mu == nu ? 1. : 0.;
      rpSecondDerivatives[munu](i, i) += 2 * Dh(nu) * rpGradient[munu](i, i) / hi;
      rpSecondDerivatives[munu](i, i) += D2h(nu) * displacements[i](mu);
      rpGradient[munu](i, i) += delta + Dh(nu) * displacements[i](mu);
    }
  }

}

  
void rWalkerHelper<rBFSlater>::OverlapWithGradient(const rDeterminant& d, 
                                                const rBFSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad,
                                                const double& ovlp) 
{
  grad.setZero();
  
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();
  std::complex<double> i(0.0, 1.0);
 
  if (schd.optimizeOrbs) {
    MatrixXd AoRi = MatrixXd::Zero(nelec, 2*norbs);
    aoValues.resize(norbs);
    
    for (int elec=0; elec<nelec; elec++) {
      schd.basis->eval(dp.coord[elec], &aoValues[0]);
      for (int orb = 0; orb<norbs; orb++) {
        if (elec < nalpha)
          AoRi(elec, orb) = aoValues[orb];
        else
          AoRi(elec, norbs+orb) = aoValues[orb];
      }
    }

    //mo gradient
    for (int mo=0; mo<nelec; mo++) {
      for (int orb=0; orb< 2*norbs; orb++) {
        std::complex<double> factor = thetaInv.row(mo) * AoRi.col(orb);
        grad[numDets + 2*orb * nelec + 2*mo] = (factor * thetaDet).real() / thetaDet.real();
        if (schd.ifComplex) grad[numDets + 2*orb * nelec + 2*mo + 1] = (i * factor * thetaDet).real() / thetaDet.real();
      }
    }
  }

  if (schd.optimizeBackflow) {
    //backflow gradient
    std::complex<double> afactor1(0., 0.);
    std::complex<double> afactor2(0., 0.);
    std::complex<double> afactorN(0., 0.);
    std::complex<double> bfactor1(0., 0.);
    std::complex<double> bfactor2(0., 0.);
    std::complex<double> bfactorN(0., 0.);
    for (int i = 0; i < nelec; i++) {
      for (int mu = 0; mu < 3; mu++) { 
        afactor1 += rTable[mu](i, i) * hValues(i) * gradaDisplacements[i][0](mu);
        afactor2 += rTable[mu](i, i) * hValues(i) * gradaDisplacements[i][1](mu);
        afactorN += rTable[mu](i, i) * hValues(i) * gradaNDisplacements[i](mu);
        bfactor1 += rTable[mu](i, i) * hValues(i) * gradbDisplacements[i][0](mu);
        bfactor2 += rTable[mu](i, i) * hValues(i) * gradbDisplacements[i][1](mu);
        bfactorN += rTable[mu](i, i) * hValues(i) * gradbNDisplacements[i](mu);
      }
    }
    grad[grad.size() - 6] = (afactorN * thetaDet).real() / thetaDet.real();
    grad[grad.size() - 5] = (bfactorN * thetaDet).real() / thetaDet.real();
    grad[grad.size() - 4] = (afactor1 * thetaDet).real() / thetaDet.real();
    grad[grad.size() - 3] = (bfactor1 * thetaDet).real() / thetaDet.real();
    grad[grad.size() - 2] = (afactor2 * thetaDet).real() / thetaDet.real();
    grad[grad.size() - 1] = (bfactor2 * thetaDet).real() / thetaDet.real();
  }
}

void rWalkerHelper<rBFSlater>::HamOverlap(const rDeterminant& d, 
                                        const rBFSlater& ref,
                                        MatrixXd& Rij,
                                        MatrixXd& RiN,
                                        Eigen::VectorBlock<VectorXd>& hamgrad)
{

}


//********************** rJASTROW ***************

rWalkerHelper<rJastrow>::rWalkerHelper(const rJastrow& cps, const rDeterminant& d,
                                       MatrixXd& Rij, MatrixXd& RiN) {
  Qmax = cps.Qmax;
  QmaxEEN = cps.QmaxEEN;

  ParamValues.setZero(cps._params.size());
  jastrowParams.setZero(cps._params.size());
  for (int i=0; i<cps._params.size(); i++) { jastrowParams(i) = cps._params[i]; }
   
  EEsameSpinIndex      = cps.EEsameSpinIndex;
  EEoppositeSpinIndex  = cps.EEoppositeSpinIndex;
  ENIndex              = cps.ENIndex;
  EENsameSpinIndex     = cps.EENsameSpinIndex;
  EENoppositeSpinIndex = cps.EENoppositeSpinIndex;
  EENNlinearIndex = cps.EENNlinearIndex;
  EENNIndex = cps.EENNIndex;

  for (int i=0; i<d.nelec; i++) {
    JastrowENValues(i, Qmax, d.coord, ParamValues, 1.0, ENIndex);

    for (int j=0; j<i; j++) {

      JastrowEEValues(i, j, Qmax, d.coord, ParamValues, 1.0, EEsameSpinIndex, 1);

      JastrowEEValues(i, j, Qmax, d.coord, ParamValues, 1.0, EEoppositeSpinIndex, 0);
      
      JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, 1.0, EENsameSpinIndex, 1);

      JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, 1.0, EENoppositeSpinIndex, 0);

    }
  }

  if (schd.fourBodyJastrow) {
    JastrowEENNinit(d.coord, N, n, gradn, lapn);
    JastrowEENNValues(N, n, ParamValues, EENNlinearIndex);
  }
 
  exponential = ParamValues.dot(jastrowParams);
}

void rWalkerHelper<rJastrow>::updateWalker(int i, Vector3d& oldcoord,
                                           const rJastrow& cps,
                                           const rDeterminant& d,
                                           MatrixXd& Rij, MatrixXd& RiN) {

  Vector3d bkp = d.coord[i]; //new coordinate

  //remove old elements
  const_cast<Vector3d&>(d.coord[i]) = oldcoord;

  JastrowENValues(i, Qmax, d.coord, ParamValues, -1.0, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;
 
    JastrowEEValues(i, j, Qmax, d.coord, ParamValues, -1.0, EEsameSpinIndex, 1);
    
    JastrowEEValues(i, j, Qmax, d.coord, ParamValues, -1.0, EEoppositeSpinIndex, 0);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, -1.0, EENsameSpinIndex, 1);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, -1.0, EENoppositeSpinIndex, 0);

  }

  //add new elements
  const_cast<Vector3d&>(d.coord[i]) = bkp;

  JastrowENValues(i, Qmax, d.coord, ParamValues, 1.0, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;

    JastrowEEValues(i, j, Qmax, d.coord, ParamValues, 1.0, EEsameSpinIndex, 1);
    
    JastrowEEValues(i, j, Qmax, d.coord, ParamValues, 1.0, EEoppositeSpinIndex, 0);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, 1.0, EENsameSpinIndex, 1);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, ParamValues, 1.0, EENoppositeSpinIndex, 0);

  }

  if (schd.fourBodyJastrow) {
    //assumes d.coord has been updated
    JastrowEENNupdate(i, d.coord, N, n, gradn, lapn);
    JastrowEENNValues(N, n, ParamValues, EENNlinearIndex);
  }

  exponential = ParamValues.dot(jastrowParams);
}


//the position of the ith electron has changed
double rWalkerHelper<rJastrow>::OverlapRatio(int i, Vector3d& coord, const rJastrow& cps,
                                             const rDeterminant &d) const
{
  double diff = 0.0;
  if (schd.fourBodyJastrow) {
    diff += JastrowEENNfactor(i, coord, d.coord, N, n, jastrowParams, EENNlinearIndex);
  }
  
  Vector3d bkp = d.coord[i];

  //new coord values
  const_cast<rDeterminant&>(d).coord[i] = coord;
 
  diff += JastrowENValue(i, Qmax, d.coord, jastrowParams, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;

    diff += JastrowEEValue(i, j, Qmax, d.coord, jastrowParams, EEsameSpinIndex, 1);
    diff += JastrowEEValue(i, j, Qmax, d.coord, jastrowParams, EEoppositeSpinIndex, 0);

    diff += JastrowEENValue(i, j, QmaxEEN, d.coord, jastrowParams, EENsameSpinIndex, 1);
    diff += JastrowEENValue(i, j, QmaxEEN, d.coord, jastrowParams, EENoppositeSpinIndex, 0);    
  }

  //old coord values
  const_cast<rDeterminant&>(d).coord[i] = bkp;
  
  diff -= JastrowENValue(i, Qmax, d.coord, jastrowParams, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;

    diff -= JastrowEEValue(i, j, Qmax, d.coord, jastrowParams, EEsameSpinIndex, 1);
    diff -= JastrowEEValue(i, j, Qmax, d.coord, jastrowParams, EEoppositeSpinIndex, 0);

      diff -= JastrowEENValue(i, j, QmaxEEN, d.coord, jastrowParams, EENsameSpinIndex, 1);
      diff -= JastrowEENValue(i, j, QmaxEEN, d.coord, jastrowParams, EENoppositeSpinIndex, 0);    
  }

  //cout << "diff: " << diff << endl;
  return exp(diff);
}


double rWalkerHelper<rJastrow>::OverlapRatioAndParamGradient(int i, Vector3d& coord, const rJastrow& cps, const rDeterminant &d, VectorXd &paramValues) const
{
  paramValues.setZero(ParamValues.size());

  if (schd.fourBodyJastrow) {
    JastrowEENNfactorVector(i, coord, d.coord, N, n, paramValues, EENNlinearIndex);
  }

  Vector3d bkp = d.coord[i]; //old coord

  //new coord values
  const_cast<rDeterminant&>(d).coord[i] = coord;

  JastrowENValues(i, Qmax, d.coord, paramValues, 1.0, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;

    JastrowEEValues(i, j, Qmax, d.coord, paramValues, 1.0, EEsameSpinIndex, 1);
    
    JastrowEEValues(i, j, Qmax, d.coord, paramValues, 1.0, EEoppositeSpinIndex, 0);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, paramValues, 1.0, EENsameSpinIndex, 1);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, paramValues, 1.0, EENoppositeSpinIndex, 0);

  }

  //old coord values
  const_cast<rDeterminant&>(d).coord[i] = bkp;

  JastrowENValues(i, Qmax, d.coord, paramValues, -1.0, ENIndex);

  for (int j=0; j<d.nelec; j++) {

    if (i == j) continue;

    JastrowEEValues(i, j, Qmax, d.coord, paramValues, -1.0, EEsameSpinIndex, 1);
    
    JastrowEEValues(i, j, Qmax, d.coord, paramValues, -1.0, EEoppositeSpinIndex, 0);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, paramValues, -1.0, EENsameSpinIndex, 1);
    
    JastrowEENValues(i, j, QmaxEEN, d.coord, paramValues, -1.0, EENoppositeSpinIndex, 0);

  }

  double exp = paramValues.dot(jastrowParams);
  //cout << "diff: " << exp << endl;
  return std::exp(exp);
}

void rWalkerHelper<rJastrow>::GradientAndLaplacian(const rDeterminant &d)
{
  GradRatio.setZero(d.nelec, 3);
  LaplaceRatio.setZero(d.nelec);
  LaplaceRatioIntermediate.setZero(d.nelec);
  
  ParamValues.setZero(jastrowParams.size());
  ParamLaplacian.setZero(d.nelec, jastrowParams.size());
  ParamGradient[0].setZero(d.nelec, jastrowParams.size());
  ParamGradient[1].setZero(d.nelec, jastrowParams.size());
  ParamGradient[2].setZero(d.nelec, jastrowParams.size());

  for (int i=0; i<d.nelec; i++) {
    JastrowEN(i, Qmax, d.coord, ParamValues, ParamGradient[0],
              ParamGradient[1], ParamGradient[2], ParamLaplacian, 1.0, ENIndex);

    for (int j=0; j<i; j++) {

      JastrowEE(i, j, Qmax, d.coord, ParamValues, ParamGradient[0],
                ParamGradient[1], ParamGradient[2], ParamLaplacian, 1.0, EEsameSpinIndex, 1);

      JastrowEE(i, j, Qmax, d.coord, ParamValues, ParamGradient[0],
                ParamGradient[1], ParamGradient[2], ParamLaplacian, 1.0, EEoppositeSpinIndex, 0);
      
      JastrowEEN(i, j, QmaxEEN, d.coord, ParamValues, ParamGradient[0],
                ParamGradient[1], ParamGradient[2], ParamLaplacian, 1.0, EENsameSpinIndex, 1);

      JastrowEEN(i, j, QmaxEEN, d.coord, ParamValues, ParamGradient[0],
                ParamGradient[1], ParamGradient[2], ParamLaplacian, 1.0, EENoppositeSpinIndex, 0);

    }
  }

  if (schd.fourBodyJastrow) {
    JastrowEENN(N, n, gradn, lapn, ParamValues, ParamGradient, ParamLaplacian, EENNlinearIndex);
  }

  exponential = ParamValues.dot(jastrowParams);
  GradRatio.col(0) = ParamGradient[0]*jastrowParams;
  GradRatio.col(1) = ParamGradient[1]*jastrowParams;
  GradRatio.col(2) = ParamGradient[2]*jastrowParams;
  LaplaceRatioIntermediate = ParamLaplacian*jastrowParams;

  for (int i=0; i<d.nelec; i++) {
    LaplaceRatio[i] = LaplaceRatioIntermediate[i] +
        pow(GradRatio(i,0), 2) +
        pow(GradRatio(i,1), 2) +
        pow(GradRatio(i,2), 2) ;
  }
}


void rWalkerHelper<rJastrow>::Gradient(int elec, Vector3d &gradRatio, const rDeterminant &d)
{
  gradRatio.setZero();

  if (schd.fourBodyJastrow) {
    JastrowEENNgradient(elec, N, n, gradn, gradRatio, jastrowParams, EENNlinearIndex);
  }

  JastrowENValueGrad(elec, Qmax, d.coord, gradRatio, jastrowParams, ENIndex);
  for (int j = 0; j < d.nelec; j++)
  {
    if (j == elec) continue;

    JastrowEEValueGrad(elec, j, Qmax, d.coord, gradRatio, jastrowParams, EEsameSpinIndex, 1);
    JastrowEEValueGrad(elec, j, Qmax, d.coord, gradRatio, jastrowParams, EEoppositeSpinIndex, 0);

    JastrowEENValueGrad(elec, j, QmaxEEN, d.coord, gradRatio, jastrowParams, EENsameSpinIndex, 1);
    JastrowEENValueGrad(elec, j, QmaxEEN, d.coord, gradRatio, jastrowParams, EENoppositeSpinIndex, 0);
  }
}


void rWalkerHelper<rJastrow>::OverlapWithGradient(const rJastrow& cps,
                                                  VectorXd& grad,
                                                  const rDeterminant& d,
                                                  const double& ovlp) const {
                                                  
  if (schd.optimizeCps) {
    for (int i=0; i<jastrowParams.size(); i++) { grad[i] = ParamValues[i]; }
    grad[EEsameSpinIndex] = 0;
    grad[EEoppositeSpinIndex] = 0;

    if (schd.noENCusp || schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { grad[ENIndex + I * Qmax] = 0.0; }
  }

}
