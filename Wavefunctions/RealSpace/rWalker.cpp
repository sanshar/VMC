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

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/math/special_functions/gamma.hpp>
#include "rWalker.h"
#include "math.h"
#include "rCorrelatedWavefunction.h"
#include "global.h"
#include "input.h"
#include "math.h"

using namespace boost;

rWalker<rJastrow, rSlater>::rWalker() {
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double>(.0,1.0); //0 mean and 1 stddev  
}

rWalker<rJastrow, rSlater>::rWalker(const rJastrow &corr, const rSlater &ref) 
{
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double> (.0,1.0); //0 mean and 1 stddev  

  initDet(ref.getHforbsA(), ref.getHforbsB());

  initR();
  initHelpers(corr, ref);
}

rWalker<rJastrow, rSlater>::rWalker(const rJastrow &corr, const rSlater &ref, const rDeterminant &pd) : d(pd)
{
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double> (.0,1.0); //0 mean and 1 stddev  
  initR();
  initHelpers(corr, ref);
}

void rWalker<rJastrow, rSlater>::initHelpers(const rJastrow &corr, const rSlater &ref)  {
  refHelper = rWalkerHelper<rSlater>(ref, d);
  corrHelper = rWalkerHelper<rJastrow>(corr, d, Rij, RiN);
}

void rWalker<rJastrow, rSlater>::initR() {
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

  RNM = MatrixXd::Zero(schd.Ncoords.size(), schd.Ncoords.size());
  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      double rij = pow( pow(schd.Ncoords[i][0] - schd.Ncoords[j][0], 2) +
                        pow(schd.Ncoords[i][1] - schd.Ncoords[j][1], 2) +
                        pow(schd.Ncoords[i][2] - schd.Ncoords[j][2], 2), 0.5);
      
      RNM(i,j) = rij;
      RNM(j,i) = rij;
    }
  }
}


rDeterminant& rWalker<rJastrow, rSlater>::getDet() {return d;}
void rWalker<rJastrow, rSlater>::readBestDeterminant(rDeterminant& d) const 
{
  if (commrank == 0) {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> d;
  }
#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, d, 0);
#endif
}


double rWalker<rJastrow, rSlater>::getDetOverlap(const rSlater &ref) const
{
  return (refHelper.thetaDet[0][0]*refHelper.thetaDet[0][1]).real();
}

/**
 * makes det based on mo coeffs 
 */
void rWalker<rJastrow, rSlater>::guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXcd& HforbsA, const Eigen::MatrixXcd& HforbsB) const 
{
/*
  auto random = std::bind(std::uniform_real_distribution<double>(-1., 1.), std::ref(generator));
  for (int i=0; i<d.nelec; i++) {
    d.coord[i][0] = random();
    d.coord[i][1] = random();
    d.coord[i][2] = random();
  }
*/
  auto random = std::bind(std::normal_distribution<double>(0.0, 1.0), std::ref(generator));
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  /*
  int i = 0;
  while (i < nelec)
  {
      for (int I = 0; I < schd.Ncharge.size(); I++)
      {
          for (int n = 0; n < schd.Ncharge[I]; n++)
          {
              Vector3d r(random(), random(), random());
              d.coord[i] = schd.Ncoords[I] + r;
              i++;

              //cout << d << endl;
          }
      }
  }
  */
  int i = 0;
  int a = 0, b = 0;
  for (int I = 0; I < schd.Ncharge.size(); I++)
  {
    for (int n = 0; n < schd.Ncharge[I];)
    {
      if (a < nalpha) {
        Vector3d r(random(), random(), random());
        d.coord[a] = schd.Ncoords[I] + r;
        //d.coord[a] = schd.Ncoords[I];
        a++;
        n++;
        i++;
  //cout << d << endl;
      }
      if (n >= schd.Ncharge[I]) continue;

      if (b < nbeta) {
        Vector3d r(random(), random(), random());
        d.coord[nalpha + b] = schd.Ncoords[I] + r;
        //d.coord[nalpha + b] = schd.Ncoords[I];
        b++;
        n++;
        i++;
  //cout << d << endl;
      }

      if (i >= nelec) break; //need this incase ion
    }
  }
  //cout << d << endl;

/*
d.coord[0] = schd.Ncoords[0] + Vector3d(0.0, 0.0, -1.0);
if (d.coord.size() == 2)
d.coord[1] = schd.Ncoords[1] + Vector3d(0.0, 0.0, 1.0);
*/

/*
  cout << "Electrons" << endl;
  std::cout << d << endl << endl;
  cout << "Basis" << endl;
  slaterBasis &basis = dynamic_cast<slaterBasis&>(*schd.basis);
  for (int i = 0; i < basis.atomicBasis.size(); i++)
  {
    std::cout << basis.atomicBasis[i] << endl;
  }
  cout << endl;
  cout << "N" << endl;
  for (int i = 0; i < schd.Ncoords.size(); i++)
  {
    std::cout << schd.Ncharge[i] << endl;
    std::cout << schd.Ncoords[i] << endl << endl;
  }
*/
}

void rWalker<rJastrow, rSlater>::initDet(const MatrixXcd& HforbsA, const MatrixXcd& HforbsB) 
{
  bool readDeterminant = false;
  char file[5000];
  sprintf(file, "BestCoordinates.txt");
  
  {
    ifstream ofile(file);
    if (ofile)
      readDeterminant = true;
  }
  if (readDeterminant)
    readBestDeterminant(d);
  else
    guessBestDeterminant(d, HforbsA, HforbsB);
}


void rWalker<rJastrow, rSlater>::updateWalker(int elec, Vector3d& coord, const rSlater& ref, const rJastrow& corr) {
  Vector3d oldCoord = d.coord[elec];
  d.coord[elec] = coord;
  for (int j=0; j<d.nelec; j++) {
    Rij(elec, j) = pow( pow(d.coord[elec][0] - d.coord[j][0], 2) +
                     pow(d.coord[elec][1] - d.coord[j][1], 2) +
                     pow(d.coord[elec][2] - d.coord[j][2], 2), 0.5);

    Rij(j,elec) = Rij(elec,j);
  }

  for (int j=0; j<schd.Ncoords.size(); j++) {
    RiN(elec, j) = pow( pow(d.coord[elec][0] - schd.Ncoords[j][0], 2) +
                     pow(d.coord[elec][1] - schd.Ncoords[j][1], 2) +
                     pow(d.coord[elec][2] - schd.Ncoords[j][2], 2), 0.5);
  }

  corrHelper.updateWalker(elec, oldCoord, corr, d, Rij, RiN);
  refHelper.updateWalker(elec, oldCoord, d, ref);

}

void rWalker<rJastrow, rSlater>::OverlapWithGradient(const rSlater &ref,
                                                    const rJastrow& cps,
                                                    VectorXd &grad) 
{
  double factor1 = 1.0;
  corrHelper.OverlapWithGradient(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows() - cps.getNumVariables());
  if (schd.optimizeOrbs == false) return;
  refHelper.OverlapWithGradient(d, ref, gradtail, factor1);
}


void rWalker<rJastrow, rSlater>::HamOverlap(const rSlater &ref,
                                           const rJastrow& cps,
                                           VectorXd &hamgrad) 
{
  //double factor1 = 1.0;
  //corrHelper.HamOverlap(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> hamtail = hamgrad.tail(hamgrad.rows()
                                                      - cps.getNumVariables());

  if (schd.optimizeOrbs == false) return;
  refHelper.HamOverlap(d, ref, Rij, RiN, hamtail);
}

void rWalker<rJastrow, rSlater>::getStep(Vector3d& coord, int elecI,
                                         double stepsize, const rSlater& ref,
                                         const rJastrow& corr,double&ovlpRatio,
                                         double& proposalProb) {
  if (schd.rStepType == SIMPLE)
    return getSimpleStep(coord, stepsize, ovlpRatio, proposalProb);
  
  else if (schd.rStepType == GAUSSIAN)
    return getGaussianStep(coord, elecI, stepsize, ovlpRatio, proposalProb);

  else if (schd.rStepType == DMC)
    return doDMCMove(coord, elecI, stepsize, ref, corr,
                      ovlpRatio, proposalProb);
  
  else if (schd.rStepType == SPHERICAL)
    return getSphericalStep(coord, elecI, stepsize, ref,
                            ovlpRatio, proposalProb);
}

void rWalker<rJastrow, rSlater>::getSimpleStep(Vector3d& coord,  double stepsize,
                                               double& ovlpRatio, double& proposalProb) {
  coord[0] = (uR()-0.5)*stepsize;
  coord[1] = (uR()-0.5)*stepsize;
  coord[2] = (uR()-0.5)*stepsize;
  proposalProb = 1.0;
  ovlpRatio = -1.0;
}

void rWalker<rJastrow, rSlater>::getGaussianStep(Vector3d& coord, int elecI, double stepsize,
                                                   double& ovlpRatio, double& proposalProb) {
  double stepx = nR(generator),
      stepy = nR(generator),
      stepz = nR(generator);
  coord[0] = stepx * stepsize;
  coord[1] = stepy * stepsize;
  coord[2] = stepz * stepsize;
  proposalProb = 1.0;
  ovlpRatio = -1.0;
}

double SphericalSteps::distance(Vector3d& r1, Vector3d& r2) {
  return pow( (r1[0]-r2[0])*(r1[0]-r2[0])+
              (r1[1]-r2[1])*(r1[1]-r2[1])+
              (r1[2]-r2[2])*(r1[2]-r2[2]), 0.5);
}

int SphericalSteps::findTheNearestNucleus(Vector3d& ri, double& riN) {
  int closestNucleus = 0;
  riN = distance(ri, schd.Ncoords[0]);
  int Natom = schd.Ncoords.size();
  
  for (int N=1; N<Natom; N++) {
    double diN = distance(ri, schd.Ncoords[N]); 

    if ( diN < riN) {
      closestNucleus = N;
      riN = diN;
    }
  }
  return closestNucleus;
}

void SphericalSteps::RotateTowards(Vector3d& R, Vector3d& ri, Vector3d& rf) {
  double rlen = pow(R[0]*R[0]+R[1]*R[1]+R[2]*R[2], 0.5);
  double rxylen = pow(R[0]*R[0]+R[1]*R[1], 0.5);
  Vector3d axis( -R[1]/rxylen, R[0]/rxylen,  0.);
  double angle = acos(R[2]/rlen);
  AngleAxis<double> aa(angle, axis);
  
  rf = aa * ri;
  return;
}


void rWalker<rJastrow, rSlater>::getGradient(int elecI, Vector3d& grad) {

  std::complex<double> DetFactor = refHelper.thetaDet[0][0] * refHelper.thetaDet[0][1], factor;
  double detgx, detgy, detgz;
  if (schd.hf == "ghf")
  {
    factor = refHelper.Gradient[0].row(elecI) * refHelper.thetaInv[0].col(elecI);
    detgx = (DetFactor * factor).real() / DetFactor.real();
    factor = refHelper.Gradient[1].row(elecI) * refHelper.thetaInv[0].col(elecI);
    detgy = (DetFactor * factor).real() / DetFactor.real();
    factor = refHelper.Gradient[2].row(elecI) * refHelper.thetaInv[0].col(elecI);
    detgz = (DetFactor * factor).real() / DetFactor.real();
  }
  else //rhf/uhf
  {
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    if (elecI < nalpha) //alpha electron
    {
      factor = refHelper.Gradient[0].row(elecI).head(nalpha) * refHelper.thetaInv[0].col(elecI);
      detgx = (DetFactor * factor).real() / DetFactor.real();
      factor = refHelper.Gradient[1].row(elecI).head(nalpha) * refHelper.thetaInv[0].col(elecI);
      detgy = (DetFactor * factor).real() / DetFactor.real();
      factor = refHelper.Gradient[2].row(elecI).head(nalpha) * refHelper.thetaInv[0].col(elecI);
      detgz = (DetFactor * factor).real() / DetFactor.real();
    }
    else //beta electron
    {
      factor = refHelper.Gradient[0].row(elecI).tail(nbeta) * refHelper.thetaInv[1].col(elecI - nalpha);
      detgx = (DetFactor * factor).real() / DetFactor.real();
      factor = refHelper.Gradient[1].row(elecI).tail(nbeta) * refHelper.thetaInv[1].col(elecI - nalpha);
      detgy = (DetFactor * factor).real() / DetFactor.real();
      factor = refHelper.Gradient[2].row(elecI).tail(nbeta) * refHelper.thetaInv[1].col(elecI - nalpha);
      detgz = (DetFactor * factor).real() / DetFactor.real();
    }
  }

  grad[0] = corrHelper.GradRatio(elecI,0) + detgx;    
  grad[1] = corrHelper.GradRatio(elecI,1) + detgy;
  grad[2] = corrHelper.GradRatio(elecI,2) + detgz;

}

double rWalker<rJastrow, rSlater>::getGradientAfterSingleElectronMove(int elecI, Vector3d& newCoord,
                                                                      Vector3d& grad,
                                                                      const rSlater& ref){


  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  vector<double>& aoValues = refHelper.aoValues;
  aoValues.resize(10*norbs, 0.0);
  schd.basis->eval_deriv2(newCoord, &aoValues[0]);

  std::complex<double> DetFactor = refHelper.thetaDet[0][0] * refHelper.thetaDet[0][1];
  double Detratio=0, gxnew=0, gynew=0, gznew=0;

  if (schd.hf == "ghf")
  {
    for (int mo=0; mo<d.nelec; mo++) {
      std::complex<double> moVal = 0, moGx=0, moGy=0, moGz=0;
      for (int j=0; j<norbs; j++) {
        int J = elecI < rDeterminant::nalpha ? j : j+norbs;
        moVal += aoValues[        j] * ref.getHforbs(0)(J, mo);
        moGx  += aoValues[norbs+  j] * ref.getHforbs(0)(J, mo);
        moGy  += aoValues[2*norbs+j] * ref.getHforbs(0)(J, mo);
        moGz  += aoValues[3*norbs+j] * ref.getHforbs(0)(J, mo);
      }
      
      Detratio += (moVal * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gxnew    += (moGx  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gynew    += (moGy  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gznew    += (moGz  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
    }
  }
  else //rhf/uhf
  {
    if (elecI < nalpha) //alpha electron
    {
      for (int mo=0; mo<nalpha; mo++) {

        std::complex<double> moVal = 0, moGx=0, moGy=0, moGz=0;
      for (int j=0; j<norbs; j++) {
        moVal += aoValues[        j] * ref.getHforbs(0)(j, mo);
        moGx  += aoValues[norbs+  j] * ref.getHforbs(0)(j, mo);
        moGy  += aoValues[2*norbs+j] * ref.getHforbs(0)(j, mo);
        moGz  += aoValues[3*norbs+j] * ref.getHforbs(0)(j, mo);
      }
      
      Detratio += (moVal * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gxnew    += (moGx  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gynew    += (moGy  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      gznew    += (moGz  * refHelper.thetaInv[0](mo, elecI) * DetFactor).real() / DetFactor.real();        
      }
    }
    else //beta electron
    {
      for (int mo=0; mo<nbeta; mo++) {

        std::complex<double> moVal = 0, moGx=0, moGy=0, moGz=0;
      for (int j=0; j<norbs; j++) {
        moVal += aoValues[        j] * ref.getHforbs(1)(j, mo);
        moGx  += aoValues[norbs+  j] * ref.getHforbs(1)(j, mo);
        moGy  += aoValues[2*norbs+j] * ref.getHforbs(1)(j, mo);
        moGz  += aoValues[3*norbs+j] * ref.getHforbs(1)(j, mo);
      }
      
      Detratio += (moVal * refHelper.thetaInv[1](mo, elecI - nalpha) * DetFactor).real() / DetFactor.real();        
      gxnew    += (moGx  * refHelper.thetaInv[1](mo, elecI - nalpha) * DetFactor).real() / DetFactor.real();        
      gynew    += (moGy  * refHelper.thetaInv[1](mo, elecI - nalpha) * DetFactor).real() / DetFactor.real();        
      gznew    += (moGz  * refHelper.thetaInv[1](mo, elecI - nalpha) * DetFactor).real() / DetFactor.real();        
      }
    }
  }
  
  gxnew /= Detratio;
  gynew /= Detratio;
  gznew /= Detratio;

  //cout << gxnew <<" update det "<<endl;
  //Do the new gx, gy, gz for the Jastrows
  double diff = 0;
  Vector3d gi, gplus, gminus; gplus.setZero();
  gminus.setZero();
  grad = corrHelper.GradRatio.row(elecI);
  grad[0] += gxnew; grad[1] += gynew; grad[2] += gznew;
  
  int Qmax = corrHelper.Qmax; VectorXd& params = corrHelper.jastrowParams;
  int QmaxEEN = corrHelper.QmaxEEN;
  int EEsameSpinIndex       = corrHelper.EEsameSpinIndex,
      EEoppositeSpinIndex   = corrHelper.EEoppositeSpinIndex,
      ENIndex               = corrHelper.ENIndex,
      EENsameSpinIndex      = corrHelper.EENsameSpinIndex,
      EENoppositeSpinIndex  = corrHelper.EENoppositeSpinIndex;

  diff -= JastrowENValueGrad(elecI, Qmax, d.coord, gminus,  params, ENIndex);
  for (int j=0; j<d.nelec; j++) {
    if (j == elecI) continue;

    diff -= JastrowEEValueGrad(elecI, j, Qmax, d.coord, gminus,  params, EEsameSpinIndex, 1);
    diff -= JastrowEEValueGrad(elecI, j, Qmax, d.coord, gminus,  params, EEoppositeSpinIndex, 0);

    diff -= JastrowEENValueGrad(elecI, j, QmaxEEN, d.coord, gminus,  params, EENsameSpinIndex, 1);
    diff -= JastrowEENValueGrad(elecI, j, QmaxEEN, d.coord, gminus,params, EENoppositeSpinIndex, 0);
  }

  Vector3d bkp = d.coord[elecI];
  d.coord[elecI] = newCoord;
  diff += JastrowENValueGrad(elecI, Qmax, d.coord, gplus,  params, ENIndex);
  for (int j=0; j<d.nelec; j++) {
    if (j == elecI) continue;
    
    diff += JastrowEEValueGrad(elecI, j, Qmax, d.coord, gplus,  params, EEsameSpinIndex, 1);
    diff += JastrowEEValueGrad(elecI, j, Qmax, d.coord, gplus,  params, EEoppositeSpinIndex, 0);

    diff += JastrowEENValueGrad(elecI, j, QmaxEEN, d.coord, gplus, params, EENsameSpinIndex, 1);
    diff += JastrowEENValueGrad(elecI, j, QmaxEEN, d.coord, gplus, params, EENoppositeSpinIndex, 0);
  }
  d.coord[elecI] = bkp;

  //cout << grad[0] - gxnew + gplus[0] - gminus[0] <<endl;
  
  grad += (gplus - gminus);
      
  double ovlpRatio = Detratio * exp(diff);

  return ovlpRatio;
}


void SphericalSteps::initializeU(Vector3d& grad, double& eta, double& a, Vector3d& di, int N) {
  Vector3d riN  = di - schd.Ncoords[N];
  double riInit = sqrt(riN.dot(riN));
  double gradR  = riN.dot(grad)/riInit; //gradient in radial direction
  double Z      = schd.Ncharge[N];   //the change of the nearest nucleus

  //solve the quadratic equation for eta
  double A = - riInit;
  double B = (Z - gradR) * riInit;
  double C =  -Z - gradR * (1 - Z * riInit);

  double temp = B*B - 4*A*C;

  //non complex and atleast one of the roots is positive
  if (temp > 0.0 && ( (-B + sqrt(temp))/2/A >= 0. ||
                      (-B - sqrt(temp))/2/A >= 0. ) ) {
    double root1 = (-B + sqrt(temp))/2/A,
        root2 = (-B - sqrt(temp))/2/A;
    eta = min(root1, root2) < 0 ? max(root1, root2) : min(root1, root2);
    a   = -Z + eta;
  }
  else if (gradR < 0) {
    eta = -gradR;
    a = 0.0;
  }
  else {
    eta = 1.0;
    a = (gradR + 1)/(1 - riInit - gradR * riInit);
  }
  
}


//the function is r^0.5 (1+ar) exp(-eta r)
double SphericalSteps::RejectionSampleR(double& eta, double& a, double& dR, double& RiN,
                                        uniformRandom& uR) {
  double rmin = RiN/dR, rmax = RiN*dR;
  double root1, root2;
  
  double umax = 0.0; //we need the max of the function r^0.5 (1 + ar) * exp(-eta r)

  {
    //take derivative of the function and set it to zero to obtain a quadratic function
    double A = - 2 * a * eta,
        B = (3 * a - 2 * eta),
        C = 1;

    root1 =  (-B + sqrt(B*B - 4*A*C))/2/A,
    root2 =  (-B - sqrt(B*B - 4*A*C))/2/A;

    if (a == 0) {
      root1 = 0.;
      root2 = 1.0/2/eta;
    }
    else if (eta == 0) {
      root1 = 0.;
      root2 = -1./3./a;
    }
    
    double urmin = abs(pow(rmin, 0.5) * (1. + a* rmin) * exp(-eta *rmin));
    double urmax = abs(pow(rmax, 0.5) * (1. + a* rmax) * exp(-eta *rmax));
    umax = max(urmin, urmax);

    if (root1 > rmin && root1 < rmax) {
      double uroot1 = abs(pow(root1, 0.5) * (1. + a* root1) * exp(-eta *root1));
      umax = max(uroot1, umax);
    }
    if (root2 > rmin && root2 < rmax) {
      double uroot2 = abs(pow(root2, 0.5) * (1. + a* root2) * exp(-eta *root2));
      umax = max(uroot2, umax);
    }    
  }
  
  while (true) {
    double rtry = rmin + (rmax - rmin) * uR();

    double urtry = abs(pow(rtry, 0.5) * (1. + a* rtry) * exp(-eta *rtry));

    if (uR() < urtry/umax) {//accept
      return rtry;
    }
  }
}

double SphericalSteps::SamplePhi(double& rif, double& thetaf, Vector3d& ri, Vector3d& gradi,
                                 int N, uniformRandom& uR) {

  return uR() * 2 * M_PI;
}

double SphericalSteps::G(double& eta, double& a, double x, double y) {
  return (boost::math::tgamma_lower(1.5, x) - boost::math::tgamma_lower(1.5,y))/pow(eta, 1.5) +
      a* (boost::math::tgamma_lower(2.5, x) - boost::math::tgamma_lower(2.5,y))/pow(eta, 2.5) ;
}


double SphericalSteps::volume(double& a, double& eta, double& Ri, double& dR, double& thetam) {
  double phitheta = (1 - cos(thetam)) * 2 * M_PI;

  if (Ri/dR < -1.0/a && -1.0/a < Ri*dR)
    return (abs(G(eta, a, Ri*dR*eta, -eta/a)) + abs(G(eta, a, -eta/a, Ri*eta/dR)))*phitheta;    
  else
    return abs(G(eta, a, Ri*dR*eta, Ri*eta/dR))*phitheta;
}

//JCP, 109, 2630
void rWalker<rJastrow, rSlater>::doDMCMove(Vector3d& coord, int elecI, double stepsize,
                                           const rSlater& ref, const rJastrow& corr, double& ovlpRatio,
                                           double& proposalProb) {

  Vector3d gradi; //gradient at initial point
  //obtain the gradient
  getGradient(elecI, gradi);

  double driftSize = pow(stepsize, 0.5);
  double stepx = nR(generator), stepy = nR(generator), stepz = nR(generator);

  double kappa = stepsize;
  double gnorm = pow(gradi[0]*gradi[0] + gradi[1]*gradi[1] + gradi[2]*gradi[2], 0.5);
  double alphainit = kappa/gnorm, deltainit = sqrt(alphainit);

  coord[0] = stepx * deltainit + alphainit * gradi[0];
  coord[1] = stepy * deltainit + alphainit * gradi[1];
  coord[2] = stepz * deltainit + alphainit * gradi[2];

  double forwardProb = exp(-(stepx*stepx + stepy*stepy + stepz*stepz)/2.)
      /pow(2*M_PI* deltainit*deltainit, 1.5);

  Vector3d newCoord = d.coord[elecI] + coord;


  //now calculate the reverse probability

  Vector3d gradInew;
  ovlpRatio = pow(getGradientAfterSingleElectronMove(elecI, newCoord, gradInew, ref),2);

  double gnormnew = pow(gradInew[0]*gradInew[0]+gradInew[1]*gradInew[1]+gradInew[2]*gradInew[2], 0.5);
  double alphanew = kappa/gnormnew, deltanew = sqrt(alphanew);//kappa1/gnormnew;
  //double alphanew = stepsize, deltanew = driftSize;
  
  //calculate the stepx/y/z needed to go back
  stepx = (-coord[0] - alphanew * gradInew[0])/deltanew;
  stepy = (-coord[1] - alphanew * gradInew[1])/deltanew;
  stepz = (-coord[2] - alphanew * gradInew[2])/deltanew;
  
  double reverseProb = exp(-(stepx*stepx + stepy*stepy + stepz*stepz)/2.)
      /pow(2*M_PI*deltanew*deltanew, 1.5);
  
  proposalProb =  reverseProb/forwardProb;
  
}

//
void rWalker<rJastrow, rSlater>::getSphericalStep(Vector3d& coord, int elecI, double stepsize,
                                                  const rSlater& ref, double& ovlpRatio,
                                                  double& proposalProb) {

  double dR = 5.0, thetaParam1 = M_PI/2; 
  Vector3d newRi; int closestNucleus; double probOfMove, probOfReverseMove;
  double volumeOfReverseMove = 1.0;

  {
    double RiN;
    closestNucleus = SphericalSteps::findTheNearestNucleus(d.coord[elecI], RiN);

    Vector3d gradi; //gradient at initial point
    getGradient(elecI, gradi);
    
    //initialize radial function U
    double eta, a;
    SphericalSteps::initializeU(gradi, eta, a, d.coord[elecI], closestNucleus);

    //sample r
    double newRiN = SphericalSteps::RejectionSampleR(eta, a, dR, RiN, uR);
    
    //sample theta
    double newTheta = acos ( 1 - uR() * (1 - cos(thetaParam1)) );
    
    //sample phi
    double newPhi = SphericalSteps::SamplePhi(newRiN, newTheta, d.coord[elecI],
                                              gradi, closestNucleus, uR);
    
    //find the new electron coordinate if RiN is the z-axis
    Vector3d riRelativeToN( newRiN * sin(newTheta) * cos(newPhi),
                            newRiN * sin(newTheta) * sin(newPhi),
                            newRiN * cos(newTheta)) ;

    Vector3d vIN = d.coord[elecI] - schd.Ncoords[closestNucleus];

    //now rotate to obtain coordinate relative to true origin
    SphericalSteps::RotateTowards(vIN, riRelativeToN, newRi);
    newRi +=  schd.Ncoords[closestNucleus];

    double Sforward = abs( pow(newRiN, -1.5) * (1 + a * newRiN) * exp(-eta * newRiN)) ;
    probOfMove = Sforward/SphericalSteps::volume(a, eta, RiN, dR, thetaParam1);
  }

  {
    double dInewNnew; //new position
    int newclosestNucleus = SphericalSteps::findTheNearestNucleus(newRi, dInewNnew);
    double Z = schd.Ncharge[newclosestNucleus];

    
    Vector3d vIoldNnew = d.coord[elecI] - schd.Ncoords[newclosestNucleus];
    Vector3d vInewNnew = newRi          - schd.Ncoords[newclosestNucleus];
    double theta = acos(vIoldNnew.dot(vInewNnew)/
                        sqrt( vIoldNnew.dot(vIoldNnew) * vInewNnew.dot(vInewNnew)));

    //first check if the move is possible
    double dIoldNnew = SphericalSteps::distance(d.coord[elecI], schd.Ncoords[newclosestNucleus]);
    
    if (theta > thetaParam1) {//move not possible
      volumeOfReverseMove = 0.0;
    }
    else if (dIoldNnew > dInewNnew*dR || dIoldNnew < dInewNnew/dR) {//move not possible
      volumeOfReverseMove = 0.0;
    }
    else {

      Vector3d gradInew;
      ovlpRatio = pow(getGradientAfterSingleElectronMove(elecI, newRi, gradInew, ref),2);

      double eta, a;
      SphericalSteps::initializeU(gradInew, eta, a, newRi, newclosestNucleus);

      double Sback = abs( pow(dIoldNnew, -1.5) * (1 + a * dIoldNnew) * exp(-eta * dIoldNnew)) ;

      volumeOfReverseMove = SphericalSteps::volume(a, eta, dInewNnew, dR, thetaParam1);
      probOfReverseMove = Sback/volumeOfReverseMove;
      
    }
  }

  coord = newRi - d.coord[elecI];

  if (abs(volumeOfReverseMove) < 1.e-20) {
    //cout << "I should not be here "<<endl;
    //cout << volumeOfReverseMove<<endl; exit(0);
    proposalProb = 0.0;
  }
  else
    proposalProb = probOfReverseMove / probOfMove;
    
}

//*******************Backflow slater**********************
rWalker<rJastrow, rBFSlater>::rWalker() {
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double>(.0,1.0); //0 mean and 1 stddev  
}

rWalker<rJastrow, rBFSlater>::rWalker(const rJastrow &corr, const rBFSlater &ref) 
{
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double> (.0,1.0); //0 mean and 1 stddev  

  initDet(ref.getHforbsA(), ref.getHforbsB());

  initR();
  initHelpers(corr, ref);
}

rWalker<rJastrow, rBFSlater>::rWalker(const rJastrow &corr, const rBFSlater &ref, const rDeterminant &pd) : d(pd)
{
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double> (.0,1.0); //0 mean and 1 stddev  
  initR();
  initHelpers(corr, ref);
}

void rWalker<rJastrow, rBFSlater>::initHelpers(const rJastrow &corr, const rBFSlater &ref)  {
  refHelper = rWalkerHelper<rBFSlater>(ref, d, Rij, RiN);
  corrHelper = rWalkerHelper<rJastrow>(corr, d, Rij, RiN);
}

void rWalker<rJastrow, rBFSlater>::initR() {
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

  RNM = MatrixXd::Zero(schd.Ncoords.size(), schd.Ncoords.size());
  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      double rij = pow( pow(schd.Ncoords[i][0] - schd.Ncoords[j][0], 2) +
                        pow(schd.Ncoords[i][1] - schd.Ncoords[j][1], 2) +
                        pow(schd.Ncoords[i][2] - schd.Ncoords[j][2], 2), 0.5);
      
      RNM(i,j) = rij;
      RNM(j,i) = rij;
    }
  }
}


rDeterminant& rWalker<rJastrow, rBFSlater>::getDet() {return d;}
void rWalker<rJastrow, rBFSlater>::readBestDeterminant(rDeterminant& d) const 
{
  if (commrank == 0) {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> d;
  }
#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, d, 0);
#endif
}


double rWalker<rJastrow, rBFSlater>::getDetOverlap(const rBFSlater &ref) const
{
  return (refHelper.thetaDet).real();
}

/**
 * makes det based on mo coeffs 
 */
void rWalker<rJastrow, rBFSlater>::guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXcd& HforbsA, const Eigen::MatrixXcd& HforbsB) const 
{
  auto random = std::bind(std::normal_distribution<double>(0.0, 1.0), std::ref(generator));
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int i = 0;
  while (i < nelec)
  {
      for (int I = 0; I < schd.Ncharge.size(); I++)
      {
          for (int n = 0; n < schd.Ncharge[I]; n++)
          {
              Vector3d r(random(), random(), random());
              int index = i / 2;
              if (i % 2 == 0) //alpha electron
                  d.coord[index] = schd.Ncoords[I] + r;
              else //beta electron
                  d.coord[index + nalpha] = schd.Ncoords[I] + r;
              i++;
          }
      }
  }
}

void rWalker<rJastrow, rBFSlater>::initDet(const MatrixXcd& HforbsA, const MatrixXcd& HforbsB) 
{
  bool readDeterminant = false;
  char file[5000];
  sprintf(file, "BestCoordinates.txt");
  
  {
    ifstream ofile(file);
    if (ofile)
      readDeterminant = true;
  }
  if (readDeterminant)
    readBestDeterminant(d);
  else
    guessBestDeterminant(d, HforbsA, HforbsB);
}


void rWalker<rJastrow, rBFSlater>::updateWalker(int elec, Vector3d& coord, const rBFSlater& ref, const rJastrow& corr) {
  Vector3d oldCoord = d.coord[elec];
  d.coord[elec] = coord;
  for (int j=0; j<d.nelec; j++) {
    Rij(elec, j) = pow( pow(d.coord[elec][0] - d.coord[j][0], 2) +
                     pow(d.coord[elec][1] - d.coord[j][1], 2) +
                     pow(d.coord[elec][2] - d.coord[j][2], 2), 0.5);

    Rij(j,elec) = Rij(elec,j);
  }

  for (int j=0; j<schd.Ncoords.size(); j++) {
    RiN(elec, j) = pow( pow(d.coord[elec][0] - schd.Ncoords[j][0], 2) +
                     pow(d.coord[elec][1] - schd.Ncoords[j][1], 2) +
                     pow(d.coord[elec][2] - schd.Ncoords[j][2], 2), 0.5);
  }

  corrHelper.updateWalker(elec, oldCoord, corr, d, Rij, RiN);
  refHelper.updateWalker(elec, oldCoord, d, Rij, RiN, ref);

}

void rWalker<rJastrow, rBFSlater>::OverlapWithGradient(const rBFSlater &ref,
                                                    const rJastrow& cps,
                                                    VectorXd &grad) 
{
  double factor1 = 1.0;
  corrHelper.OverlapWithGradient(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows() - cps.getNumVariables());
  //if (schd.optimizeOrbs == false) return;
  refHelper.OverlapWithGradient(d, ref, gradtail, factor1);
}


void rWalker<rJastrow, rBFSlater>::HamOverlap(const rBFSlater &ref,
                                           const rJastrow& cps,
                                           VectorXd &hamgrad) 
{
  //double factor1 = 1.0;
  //corrHelper.HamOverlap(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> hamtail = hamgrad.tail(hamgrad.rows()
                                                      - cps.getNumVariables());

  if (schd.optimizeOrbs == false) return;
  refHelper.HamOverlap(d, ref, Rij, RiN, hamtail);
}

void rWalker<rJastrow, rBFSlater>::getStep(Vector3d& coord, int elecI,
                                         double stepsize, const rBFSlater& ref,
                                         const rJastrow& corr,double&ovlpRatio,
                                         double& proposalProb) {
  if (schd.rStepType == SIMPLE)
    return getSimpleStep(coord, stepsize, ovlpRatio, proposalProb);
  
  else if (schd.rStepType == GAUSSIAN)
    return getGaussianStep(coord, elecI, stepsize, ovlpRatio, proposalProb);

  else if (schd.rStepType == DMC)
    return doDMCMove(coord, elecI, stepsize, ref, corr,
                      ovlpRatio, proposalProb);
  
  else if (schd.rStepType == SPHERICAL)
    return getSphericalStep(coord, elecI, stepsize, ref,
                            ovlpRatio, proposalProb);
}

void rWalker<rJastrow, rBFSlater>::getSimpleStep(Vector3d& coord,  double stepsize,
                                               double& ovlpRatio, double& proposalProb) {
  coord[0] = (uR()-0.5)*stepsize;
  coord[1] = (uR()-0.5)*stepsize;
  coord[2] = (uR()-0.5)*stepsize;
  proposalProb = 1.0;
  ovlpRatio = -1.0;
}

void rWalker<rJastrow, rBFSlater>::getGaussianStep(Vector3d& coord, int elecI, double stepsize,
                                                   double& ovlpRatio, double& proposalProb) {
  double stepx = nR(generator),
      stepy = nR(generator),
      stepz = nR(generator);
  coord[0] = stepx * stepsize;
  coord[1] = stepy * stepsize;
  coord[2] = stepz * stepsize;
  proposalProb = 1.0;
  ovlpRatio = -1.0;
}

void rWalker<rJastrow, rBFSlater>::doDMCMove(Vector3d& coord, int elecI, double stepsize,
                                           const rBFSlater& ref, const rJastrow& corr, double& ovlpRatio,
                                           double& proposalProb) 
{

}

void rWalker<rJastrow, rBFSlater>::getSphericalStep(Vector3d& coord, int elecI, double stepsize,
                                                  const rBFSlater& ref, double& ovlpRatio,
                                                  double& proposalProb) 
{

}

void rWalker<rJastrow, rBFSlater>::getGradient(int elecI, Vector3d& grad) 
{

}

double rWalker<rJastrow, rBFSlater>::getGradientAfterSingleElectronMove(int elecI, Vector3d& newCoord,
                                                                      Vector3d& grad,
                                                                      const rBFSlater& ref)
{
  return 0.;
}
