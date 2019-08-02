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
#ifndef EvalETranscorrelated_HEADER_H
#define EvalETranscorrelated_HEADER_H
#include <Eigen/Dense>
#include "igl/slice.h"
#include <vector>
#include "Determinants.h"
#include "rDeterminants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "linearMethod.h"
#include "global.h"
#include "Deterministic.h"
#include "ContinuousTime.h"
#include "Metropolis.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>
#include <stan/math.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <complex>
#include <cmath>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;
using namespace boost;
//using namespace stan::math;
using namespace std::complex_literals;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}

//term is orb1^dag orb2
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int norbs,
                           const VectorXd&JA) {
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  double factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;
}

//term is orb1^dag orb2^dag orb3 orb4
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int orb3,
                           int orb4,
                           int norbs,
                           const VectorXd&JA) {

  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  double factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                       -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
}



//N_orbn N_orbm orb1^dag orb2^dag orb3 orb4
complex<double> getResidue(MatrixXcd& rdm, int orbn, int orbm, int orb1, int orb2, int orb3, int orb4) {

  vector<int> rows = {orbm, orbn, orb3, orb4},
              cols = {orbm, orbn, orb2, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 4), colVec(&cols[0], 4);

  complex<double> contribution ;
  MatrixXcd rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();

  if (orbm == orb2) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbn == orb2) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbn == orb2 && orbm == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1 && orbm == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  return contribution;
}


//N_orbn orb1^dag orb2^dag orb3 orb4
complex<double> getResidue(MatrixXcd& rdm, int orbn, int orb1, int orb2, int orb3, int orb4) {

  vector<int> rows = {orbn, orb3, orb4},
              cols = {orbn, orb2, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 3), colVec(&cols[0], 3);

  complex<double> contribution ;
  MatrixXcd rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb1; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb2; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }

  return contribution;
}

//N_orbn N_orbm orb1^dag orb2
complex<double> getResidue(const MatrixXcd& rdm, int orbn, int orbm, int orb1, int orb2) {
  
  vector<int> rows = {orbn, orbm, orb2},
              cols = {orbn, orbm, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 3), colVec(&cols[0], 3);

  complex<double> contribution;
  MatrixXcd rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    MatrixXcd rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  return contribution;
}



//N_orbn orb1^dag orb2
complex<double> getResidue(MatrixXcd& rdm, int orbn, int orb1, int orb2) {

  vector<int> rows = {orbn, orb2},
              cols = {orbn, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 2), colVec(&cols[0], 2);

  complex<double> contribution ;
  MatrixXcd rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb1) {
    contribution += rdm(orb2, orbn);
  }

  return contribution;
}

//here T can be stan::math::var or double depending on whether we are using automatic differentiation or not
struct Residuals {
  int norbs, nalpha, nbeta;
  MatrixXcd bra;
  vector<MatrixXcd > ket;
  vector<complex<double> > coeffs;
  double E0;
  
  Residuals(int _norbs, int _nalpha, int _nbeta,
            MatrixXcd& _bra,
            vector<MatrixXcd >& _ket,
            vector<complex<double> >& _coeffs) : norbs(_norbs),
                                                 nalpha(_nalpha),
                                                 nbeta(_nbeta),
                                                 bra(_bra),
                                                 ket(_ket),
                                                 coeffs(_coeffs){};


  int operator() (const VectorXd& JA, VectorXd& residue) const
  {
    //calculate <bra|P|ket> and
    double detovlp = 0.0;
    MatrixXd mfRDM(2*norbs, 2*norbs); mfRDM.setZero();
    
    for (int g = 0; g<ket.size(); g++) {
      MatrixXcd S = bra.adjoint()*ket[g];
      complex<double> Sdet = S.determinant();
      detovlp += (S.determinant() * coeffs[g]).real();
      mfRDM += ((ket[g] * S.inverse())*bra.adjoint() * coeffs[g] * Sdet).real(); 
    }
    mfRDM = mfRDM/detovlp;
    
    double Energy = 0.;
    VectorXd intermediateResidue(2*norbs*(2*norbs+1)/2); intermediateResidue.setZero();

    for (int g = 0; g<ket.size(); g++)  {
      Energy += (getResidueSingleKet(JA, intermediateResidue,
                                     detovlp, coeffs[g], bra, ket[g]) * coeffs[g]).real();
    }    

    for (int i=0; i<2*norbs; i++) {
      for (int j=0; j<=i; j++) {
        int index = i*(i+1)/2+j;
        if (i == j) 
          residue(i * (i+1)/2 + j) = intermediateResidue(i*(i+1)/2+j) - (Energy)*mfRDM(i,i);
        else
          residue(i * (i+1)/2 + j) = intermediateResidue(i*(i+1)/2+j) - (Energy)*(mfRDM(j,j)*mfRDM(i,i) - mfRDM(j,i)*mfRDM(i,j));
      }
    }

    const_cast<double&>(this->E0) = Energy;
    return 0;
  }
  
  complex<double> getResidueSingleKet (const VectorXd& JA,
                              VectorXd& residue, double detovlp, complex<double> coeff,
                              const MatrixXcd& bra, const MatrixXcd& ket) const
  {
    MatrixXcd LambdaD = bra, LambdaC = ket; //just initializing  
    MatrixXcd S = bra.adjoint()*ket;

    DiagonalXd diagcre(2*norbs),
        diagdes(2*norbs);
    
    complex<double> Energy = 0.0;
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
        LambdaD = diagdes*ket;
        LambdaC = diagcre*bra;
        S = LambdaC.adjoint()*LambdaD;

        factor *= S.determinant()/detovlp;
        MatrixXcd rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
        Energy += rdm(orb2,orb1) * integral * factor;

        complex<double> res;
        for (int orbn = 0; orbn < 2*norbs; orbn++) {
          for (int orbm = 0; orbm < orbn; orbm++) {
            res = getResidue(rdm, orbn, orbm, orb1, orb2);
            residue(orbn*(orbn+1)/2 + orbm) += (res * factor * integral * coeff).real();
          }
          res = getResidue(rdm, orbn, orb1, orb2);
          residue(orbn*(orbn+1)/2 + orbn) += (res * factor * integral * coeff).real();
        }
      }
    
    
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < orb1; orb2++) {
        for (int orb3 = 0; orb3 < 2*norbs; orb3++)
          for (int orb4 = 0; orb4 < orb3; orb4++) {
            
            int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
            int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
            int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
            int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
            double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
            if (abs(integral) < schd.epsilon) continue;
            
            complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
            LambdaD = diagdes*ket;
            LambdaC = diagcre*bra;
            S = LambdaC.adjoint()*LambdaD;

            MatrixXcd rdm = ((LambdaD * S.inverse()) * LambdaC.adjoint());
            complex<double> rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
            factor *= S.determinant()/detovlp;
            Energy += rdmval * integral * factor;

            complex<double> res;
            for (int orbn = 0; orbn < 2*norbs; orbn++) {
              for (int orbm = 0; orbm < orbn; orbm++) {
                res = getResidue(rdm, orbn, orbm, orb1, orb2, orb3, orb4);
                residue(orbn*(orbn+1)/2 + orbm) += (res*factor*integral * coeff).real();
              }
              res = getResidue(rdm, orbn, orb1, orb2, orb3, orb4);
              residue(orbn*(orbn+1)/2 + orbn) += (res*factor*integral * coeff).real();
            }
            
          }
      }

    return Energy;
  }


  //obviously the energy should be real
  double energy (const VectorXd& JA) const
  {
    //calculate <bra|P|ket> and
    complex<double> detovlp = 0.0;
    for (int g = 0; g<ket.size(); g++) {
      MatrixXcd S = bra.adjoint()*ket[g];
      detovlp += (S.determinant() * coeffs[g]).real();
    }


    double Energy = 0.0;
    for (int g = 0; g<ket.size(); g++) 
      Energy += (energyContribution(JA, detovlp.real(), bra, ket[g]) * coeffs[g]).real();

    return Energy;
  }
  
  complex<double> energyContribution (const VectorXd& JA, double detovlp,
                                      const MatrixXcd& bra, const MatrixXcd& ket) const
  {

    MatrixXcd LambdaC, LambdaD, S;
    DiagonalXd diagcre(2*norbs),
        diagdes(2*norbs);

    complex<double> Energy = 0.0;
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
        LambdaD = diagdes*ket;
        LambdaC = diagcre*bra;
        S = LambdaC.adjoint()*LambdaD;

        factor *= S.determinant()/detovlp;
        //**don't need to calculate the entire RDM, should make it more efficient
        MatrixXcd rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
        Energy += rdm(orb2,orb1) * integral * factor;
      }
    
    
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < orb1; orb2++) {
        for (int orb3 = 0; orb3 < 2*norbs; orb3++)
          for (int orb4 = 0; orb4 < orb3; orb4++) {
            
            int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
            int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
            int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
            int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
            double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
            if (abs(integral) < schd.epsilon) continue;
            
            complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
            LambdaD = diagdes*ket;
            LambdaC = diagcre*bra;
            S = LambdaC.adjoint()*LambdaD;

            //**don't need to calculate the entire RDM, should make it more efficient
            MatrixXcd rdm = ((LambdaD * S.inverse()) * LambdaC.adjoint());
            complex<double> rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
            factor *= S.determinant()/detovlp;
            Energy += rdmval * integral * factor;
          }
      }

    return Energy;
  }

};


template <typename Wfn, typename Walker>
class getTranscorrelationWrapper
{
 public:
  Wfn &w;
  Walker &walk;
  int stochasticIter;
  bool ctmc;
  getTranscorrelationWrapper(Wfn &pw, Walker &pwalk, int niter, bool pctmc) : w(pw), walk(pwalk)
  {
    stochasticIter = niter;
    ctmc = pctmc;
  };

  double getTransGradient(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    //using VarType = stan::math::var ;
    //using VarType = double;
    using VarType = std::complex<double>;
    
    w.updateVariables(vars);
    w.initWalker(walk);
    
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    
    VectorXd residue(2*norbs*(2*norbs+1)/2); residue.setZero();
    VectorXd JA     (2*norbs*(2*norbs+1)/2);
    VarType Energy;
    
    MatrixXcd bra = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta);
    MatrixXcd ket = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta);

    //grid to project out the Sz quantum number
    double Sz = 0;
    int ngrid = 10;
    complex<double> iImag(0, 1.0);
    vector<complex<double> > coeffs(ngrid*2, 0.0);
    vector< MatrixXcd > ketvec(ngrid*2, ket);
    //vector<complex<double> > coeffs(ngrid*1, 0.0);
    //vector< MatrixXcd > ketvec(ngrid*1, ket);

    for (int g = 0; g<ngrid; g++) {
      DiagonalMatrix<complex<double>, Dynamic> phi(2*norbs);
      double angle = g*2*M_PI/ngrid;
      for (int i=0; i<norbs; i++) {
        phi.diagonal()[i]       = exp(iImag*angle);
        phi.diagonal()[i+norbs] = exp(-iImag*angle);
      }
      ketvec[2*g] = phi * ketvec[g];
      coeffs[2*g] = exp(-iImag*angle*Sz)/ngrid;

      ketvec[2*g+1] = (phi * ketvec[g]).conjugate();
      coeffs[2*g+1] = exp(iImag*angle*Sz)/ngrid;
    }
    
    MatrixXd& Jtmp = w.getCorr().SpinCorrelator;
    for (int i=0; i<2*norbs; i++) {
      int I = (i/2) + (i%2)*norbs;
      for (int j=0; j<i; j++) {
        int J = (j/2) + (j%2)*norbs;
        
        JA(index(J, I)) = log(Jtmp(i, j))/2.0;
      }
      JA(index(I, I)) = log(Jtmp(i,i));///2;
    }
    
    Residuals residual(norbs, nalpha, nbeta, bra, ketvec, coeffs);
    std::cout << "Initial Energy: " <<residual.energy(JA) <<endl;

    double norm = 10.; 
    int iter = 0;
    HybridNonLinearSolver<Residuals > solver(residual);

    solver.solveNumericalDiffInit(JA);
    while(norm > 1.e-6) {
      iter++;
      int info = solver.solveNumericalDiffOneStep(JA);
      residual(JA, residue);
      norm = residue.norm();
      std::cout << format("%5i   %14.8f   %14.6f \n") %(iter) %(residual.E0) %(norm);
    }
    exit(0);
    /*
    int iter = 0, niter = 10;
    while (true) {
      //Solve the Jastrow parameters
      Residuals<VarType> residual(norbs, nalpha, nbeta, bra, ketvec );
      HybridNonLinearSolver<Residuals<VarType> > solver(residual);
      int info = solver.solveNumericalDiff(JA);
      
      if (info != 1) {
        cout << "Jastrow optimizer didn't converge"<<endl;
        exit(0);
      }
      //std::cout << "iter count: " << solver.iter << std::endl;
      //std::cout << "return status: " << info << std::endl;
      //residual(JA, residue);
      std::cout << "Energy: " <<residual.energy(JA) <<endl;
      std::cout << "Error: " <<residue.norm() <<endl;
      std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      
      //solve the lambda equations
    
      
      }
    */
    w.writeWave();
    exit(0);
    return 1.0;
  }
  
};




#endif

