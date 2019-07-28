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

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;
using namespace boost;
//using namespace stan::math;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}

//term is orb1^dag orb2
template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int norbs,
                      const Matrix<T, Dynamic, 1>&JA) {
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  T factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;
}

//term is orb1^dag orb2^dag orb3 orb4
template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int orb3,
                      int orb4,
                      int norbs,
                      const Matrix<T, Dynamic, 1>&JA) {

  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  T factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                      -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
}



//N_orbn N_orbm orb1^dag orb2^dag orb3 orb4
template<typename T>
T getResidue(Matrix<T, Dynamic, Dynamic>& rdm, int orbn, int orbm, int orb1, int orb2, int orb3, int orb4) {

  vector<int> rows = {orbm, orbn, orb3, orb4},
              cols = {orbm, orbn, orb2, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 4), colVec(&cols[0], 4);

  T contribution = 0;
  Matrix<T, Dynamic, Dynamic> rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();

  if (orbm == orb2) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbn == orb2) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbn == orb2 && orbm == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 3);
    Eigen::Map<VectorXi> colVec(&cols[0], 3);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1 && orbm == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  return contribution;
}


//N_orbn orb1^dag orb2^dag orb3 orb4
template<typename T>
T getResidue(Matrix<T, Dynamic, Dynamic>& rdm, int orbn, int orb1, int orb2, int orb3, int orb4) {

  vector<int> rows = {orbn, orb3, orb4},
              cols = {orbn, orb2, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 3), colVec(&cols[0], 3);

  T contribution = 0;
  Matrix<T, Dynamic, Dynamic> rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb1; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  if (orbn == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb2; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }

  return contribution;
}

//N_orbn N_orbm orb1^dag orb2
template<typename T>
T getResidue(Matrix<T, Dynamic, Dynamic>& rdm, int orbn, int orbm, int orb1, int orb2) {

  vector<int> rows = {orbn, orbm, orb2},
              cols = {orbn, orbm, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 3), colVec(&cols[0], 3);

  T contribution = 0;
  Matrix<T, Dynamic, Dynamic> rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution += rdmval.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Eigen::Map<VectorXi> rowVec(&rows[0], 2);
    Eigen::Map<VectorXi> colVec(&cols[0], 2);
    Matrix<T, Dynamic, Dynamic> rdmval;
    igl::slice(rdm, rowVec, colVec, rdmval);
    contribution -= rdmval.determinant();
  }
  return contribution;
}



//N_orbn orb1^dag orb2
template<typename T>
T getResidue(Matrix<T, Dynamic, Dynamic>& rdm, int orbn, int orb1, int orb2) {

  vector<int> rows = {orbn, orb2},
              cols = {orbn, orb1};
  Eigen::Map<VectorXi> rowVec(&rows[0], 2), colVec(&cols[0], 2);

  T contribution = 0;
  Matrix<T, Dynamic, Dynamic> rdmval;
  igl::slice(rdm, rowVec, colVec, rdmval);
  contribution =  rdmval.determinant();
  
  if (orbn == orb1) {
    contribution += rdm(orb2, orbn);
  }

  return contribution;
}

template <class T>
struct Residuals {
  using MatrixXt = Matrix<T, Dynamic, Dynamic>;
  using VectorXt = Matrix<T, Dynamic, 1>;

  int norbs, nalpha, nbeta;
  MatrixXt bra;
  vector<MatrixXt > ket;
  VectorXt JA;
  VectorXt lambda;
  
  Residuals(int _norbs, int _nalpha, int _nbeta,
            MatrixXt& _bra,
            vector<MatrixXt >& _ket) : norbs(_norbs),
                                       nalpha(_nalpha),
                                       nbeta(_nbeta),
                                       bra(_bra),
                                       ket(_ket) {};
  
  int operator() (const VectorXt& JA,
                  VectorXt& residue) const
  {
    residue.setZero();
    
    MatrixXt LambdaD = bra, LambdaC = ket[0]; //just initializing  
    MatrixXt S = bra.transpose()*ket[0];
    T detovlp = S.determinant(); 

    MatrixXt mfRDM = (ket[0] * S.inverse()) * bra.transpose();

    DiagonalMatrix<T, Dynamic> diagcre(2*norbs),
        diagdes(2*norbs);

    T Energy = 0.0;
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        T factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
        LambdaD = diagdes*ket[0];
        LambdaC = diagcre*bra;
        S = LambdaC.transpose()*LambdaD;

        factor *= S.determinant()/detovlp;
        MatrixXt rdm = (LambdaD * S.inverse())*LambdaC.transpose();
        Energy += rdm(orb2,orb1) * integral * factor;

        T res;
        for (int orbn = 0; orbn < 2*norbs; orbn++) {
          for (int orbm = 0; orbm < orbn; orbm++) {
            res = getResidue(rdm, orbn, orbm, orb1, orb2);
            residue(orbn*(orbn+1)/2 + orbm) += res * factor * integral;
          }
          res = getResidue(rdm, orbn, orb1, orb2);
          residue(orbn*(orbn+1)/2 + orbn) += res * factor * integral;
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
            
            T factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
            LambdaD = diagdes*ket[0];
            LambdaC = diagcre*bra;
            S = LambdaC.transpose()*LambdaD;

            MatrixXt rdm = ((LambdaD * S.inverse()) * LambdaC.transpose());
            T rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
            factor *= S.determinant()/detovlp;
            Energy += rdmval * integral * factor;

            T res;
            for (int orbn = 0; orbn < 2*norbs; orbn++) {
              for (int orbm = 0; orbm < orbn; orbm++) {
                res = getResidue(rdm, orbn, orbm, orb1, orb2, orb3, orb4);
                residue(orbn*(orbn+1)/2 + orbm) += res*factor*integral;
              }
              res = getResidue(rdm, orbn, orb1, orb2, orb3, orb4);
              residue(orbn*(orbn+1)/2 + orbn) += res*factor*integral;
            }
            
          }
      }

    for (int i=0; i<2*norbs; i++) {
      for (int j=0; j<=i; j++) {
        int index = i*(i+1)/2+j;
        if (i == j) 
          residue(i * (i+1)/2 + j) -= (Energy)*mfRDM(i,i);
        else
          residue(i * (i+1)/2 + j) -= (Energy)*(mfRDM(j,j)*mfRDM(i,i) - mfRDM(j,i)*mfRDM(i,j));
      }
    }

    
    return 0;
  }

  T energy (const VectorXt& JA) const
  {
    MatrixXt LambdaD = bra, LambdaC = ket[0]; //just initializing  
    MatrixXt S = bra.transpose()*ket[0];
    T detovlp = S.determinant(); 

    MatrixXt mfRDM = (ket[0] * S.inverse()) * bra.transpose();

    DiagonalMatrix<T, Dynamic> diagcre(2*norbs),
        diagdes(2*norbs);

    T Energy = 0.0;
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        T factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
        LambdaD = diagdes*ket[0];
        LambdaC = diagcre*bra;
        S = LambdaC.transpose()*LambdaD;

        factor *= S.determinant()/detovlp;
        MatrixXt rdm = (LambdaD * S.inverse())*LambdaC.transpose();
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
            
            T factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
            LambdaD = diagdes*ket[0];
            LambdaC = diagcre*bra;
            S = LambdaC.transpose()*LambdaD;

            MatrixXt rdm = ((LambdaD * S.inverse()) * LambdaC.transpose());
            T rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
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
    using VarType = double;
    
    w.updateVariables(vars);
    w.initWalker(walk);

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;

    /*
    stddev = 0.0;
    rt = 1.0;


    std::vector<Determinant> allDets;
    vector<vector<int> > alldetvec;
    comb(2*norbs, nalpha+nbeta, alldetvec);
    for (int a=0; a<alldetvec.size(); a++) {
      Determinant d;
      for (int i=0; i<alldetvec[a].size(); i++)
        d.setocc(alldetvec[a][i], true);
      allDets.push_back(d);
    }
    
    double Energy2 = 0.0;
    double denominator = 0.0;
    workingArray work;
    
    for (int i = commrank; i < allDets.size(); i += commsize)
    {
      double ovlp, Eloc;
      w.initWalker(walk, allDets[i]);
      w.HamAndOvlp(walk, ovlp, Eloc, work, false);  

      double corrovlp = w.getCorr().Overlap(walk.d),
          slaterovlp = walk.getDetOverlap(w.getRef());

      cout << allDets[i]<<"  "<<ovlp<<endl;
      Energy2 += slaterovlp/corrovlp * Eloc*ovlp;
      denominator += slaterovlp*slaterovlp;
    }
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &(Energy2), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(denominator), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    if (commrank == 0) {
      cout << Energy2<<"  "<<denominator <<endl;
      cout << Energy2/denominator <<endl;
    }
    */
    
    Matrix<VarType, Dynamic, 1> residue(2*norbs*(2*norbs+1)/2); residue.setZero();
    Matrix<VarType, Dynamic, 1> JA     (2*norbs*(2*norbs+1)/2);
    VarType Energy;
    
    Matrix<VarType, Dynamic, Dynamic> bra = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta).real();
    Matrix<VarType, Dynamic, Dynamic> ket = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta).real();
    vector< Matrix<VarType, Dynamic, Dynamic> > ketvec(1,ket);
    
    MatrixXd& Jtmp = w.getCorr().SpinCorrelator;
    for (int i=0; i<2*norbs; i++) {
      int I = (i/2) + (i%2)*norbs;
      for (int j=0; j<i; j++) {
        int J = (j/2) + (j%2)*norbs;
        
        JA(index(J, I)) = log(Jtmp(i, j))/2.0;
      }
      JA(index(I, I)) = log(Jtmp(i,i));///2;
    }

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
    

    /*
    VectorXd res(2*norbs*(2*norbs+1)/2);
    double residueNorm = 10;
    
    for (int iter = 0; iter<100 && residueNorm > 1.e-5; iter++) {
      residue = residual(bra, ket, JA, Energy);
      residueNorm = residue.norm().val();
      cout << Energy+coreE<<"  "<<residue.norm()<<"  "<<endl;
      Matrix<double, Dynamic, Dynamic> J(residue.size(), JA.size());

      //auto oldresidue = residue;
      //JA(1) += 1.e-3;
      //residue = residual(bra, ket, JA, Energy);
      //cout << (residue - oldresidue)/1.e-3<<endl<<" fd "<<endl;
      for (int i = 0; i < residue.size(); ++i) {
        res(i) = residue(i).val();
        if (i > 0) stan::math::set_zero_all_adjoints();
        residue(i).grad();
        for (int j = 0; j < JA.size(); ++j) {
          J(i,j) = JA(j).adj();
        }
      }

      //cout << J <<endl;
      FullPivLU<MatrixXd> lu(J); //cout << lu.rank()<<endl;
      //VectorXd update= - J.colPivHouseholderQr().solve(res);
      VectorXd update= - lu.solve(res);

      auto Jtemp = JA;
      vector<double> scale = {0.1, 0.05, 0.02, 0.01, 0.005};

      double error; double scaleToUse = scale[0];
      /*
      Jtemp += scale[0]*update;
      error = residual(bra, ket, Jtemp, Energy).norm().val();
      Jtemp -= scale[0]*update;
      
      for (int s=1; s<scale.size(); s++) {
        Jtemp += scale[s]*update;
        double currentError = residual(bra, ket, Jtemp, Energy).norm().val();
        cout << currentError<<"  "<<error<<endl;
        if (currentError < error) {
          error = currentError;
          scaleToUse = scale[s];
        }
        Jtemp -= scale[s]*update;
      }
      cout << scaleToUse<<endl;
      
      scaleToUse = 0.1;
      
      JA += scaleToUse*update;
    }
    /*
    for (int iter=0; iter<100; iter++) {
      getTransEnergyResidue(w, walk, E0, residue, grad);

      cout << E0+coreE<<"  "<<residue.norm()<<endl;
      for (int o1=0; o1<2*norbs; o1++)
        for (int o2=0; o2<=o1; o2++) {
          int jo1 = (o1%norbs)*2 + (o1/norbs),
              jo2 = (o2%norbs)*2 + (o2/norbs);
          J(jo1, jo2)  *= exp(-residue(o1*(o1+1)/2+o2)/grad(o1*(o1+1)/2+o2));
        }
    }
    */
    w.writeWave();
    exit(0);
    return 1.0;
  };

};




#endif

