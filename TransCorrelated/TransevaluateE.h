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
#pragma once

#include "optimizer.h"
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "rDeterminants.h"
#include "global.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <complex>
#include <cmath>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include "DirectJacobian.h"
#include "Residuals.h"
#include "ResidualsGJ.h"
#include "CorrelatedWavefunction.h"
#include "Complex.h"

#include "LevenbergHelper.h"
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>
#include "stan/math.hpp"
#include "amsgrad.h"
//#include "ceres/ceres.h"
//#include "glog/logging.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;
using namespace boost;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

template<typename Wfn>
void saveWave(VectorXd& variables, Wfn& w,
              vector<pair<int, int>>& Jmap,
              VectorXd& JRed) {
  
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  int nelec = nalpha + nbeta;
  
  int nJastrowVars = Jmap.size(); 
  int nOrbitalVars = (2*norbs-nelec)*(nelec);
  VectorXd JnonRed = variables.block(0,0,nJastrowVars,1);
  VectorXd braVars = variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1);
  VectorXd JA(2*norbs*(2*norbs+1)/2);
  
  JastrowFromRedundantAndNonRedundant(JA, JRed, JnonRed, Jmap);
  fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);

  MatrixXcd hforbsa = w.getRef().HforbsA;
  MatrixXcd&& bra = fillWfnOrbs(w.getRef().HforbsA, braVars);
  w.getRef().HforbsA.block(0,0,2*norbs, nelec) = 1.*bra;
  w.writeWave();
  w.getRef().HforbsA = hforbsa;
  
  //variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1).setZero();
}



template <typename Wfn>
class getTranscorrelationWrapper
{
 public:
  Wfn &w;
  getTranscorrelationWrapper(Wfn &pwr) : w(pwr)
  {};


  double optimizeWavefunctionGJ()
  {
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));
    //using T = stan::math::var;
    using T = double;
    using complexT = complex<T>;
    //using complexT = Complex<T>;
    using MatrixXcT = Matrix<complexT, Dynamic, Dynamic>;
    using VectorXcT = Matrix<complexT, Dynamic, 1>;
    using MatrixXT = Matrix<T, Dynamic, Dynamic>;
    using VectorXT = Matrix<T, Dynamic, 1>;
    
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    int nelec = nalpha + nbeta;
    
    int nJastrowVars = norbs;
    int nOrbitalVars = (2*norbs-nelec)*(nelec);
    VectorXT variables(nJastrowVars + 2*nOrbitalVars);
    VectorXT Jastrow(nJastrowVars), Lambda(nJastrowVars);
    //Jastrow = VectorXT::Random(norbs);//just for debugging
    for (int i=0; i<nJastrowVars; i++) {
      Lambda[i] = 0.0;
      Jastrow[i] = 0.01;
    }
    //Lambda.setZero(); Jastrow.setZero();

    MatrixXcd& hforbs = w.getRef().HforbsA;
    
    MatrixXcT bra(2*norbs, nelec);
    for (int i=0; i<2*norbs; i++)
      for (int j=0; j<nelec; j++) {
        bra(i,j).real(hforbs(i,j).real());
        bra(i,j).imag(hforbs(i,j).imag());
      }
    int ngrid = 4; //FOR THE SZ PROJECTOR


    GetResidualGJ<T, complexT> resGJ(ngrid);
    T E = resGJ.getLagrangian(bra, Jastrow, Lambda);
    cout << E<<"  "<<getTime()-startofCalc<<endl;

    /*
    grad(E.vi_);
    for (int i=0; i<2*norbs; i++)
      for (int j=0; j<nelec; j++)
        cout << bra(i,j)<<"  "<<bra(i,j).real().adj()<<"  "<<bra(i,j).imag().adj()<<endl;
    cout << "***** "<<getTime()-startofCalc<<endl;
    */
  }
  
  double optimizeWavefunction()
  {
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));
    //using T = stan::math::var;
    using T = double;
    using complexT = complex<T>;
    //using complexT = Complex<T>;
    using MatrixXcT = Matrix<complexT, Dynamic, Dynamic>;
    using VectorXcT = Matrix<complexT, Dynamic, 1>;
    using MatrixXT = Matrix<T, Dynamic, Dynamic>;
    using VectorXT = Matrix<T, Dynamic, 1>;
    
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    int nelec = nalpha + nbeta;
    
    vector<pair<int, int>> NonRedundantMap;
    ConstructRedundantJastrowMap(NonRedundantMap);
    int nJastrowVars = NonRedundantMap.size(); //4k redundancies
    int nOrbitalVars = (2*norbs-nelec)*(nelec);
    VectorXT variables(nJastrowVars + 2*nOrbitalVars);

    VectorXT JA(2*norbs*(2*norbs+1)/2); 
    VectorXT JRed(2*norbs*(2*norbs+1)/2 - nJastrowVars), JnonRed(nJastrowVars);

    fillJastrowfromWfn(w.getCorr().SpinCorrelator, JA);
    RedundantAndNonRedundantJastrow(JA, JRed, JnonRed, NonRedundantMap);
    JnonRed = VectorXT::Random(norbs);//just for debugging
    
    VectorXT braVars(2* nOrbitalVars); braVars.setZero();
    int ngrid = 4; //FOR THE SZ PROJECTOR


    MatrixXcT hforbs(2*norbs, 2*norbs);
    hforbs = w.getRef().HforbsA;

    //calculates orbital, Jastrow Residue
    GetResidual<T, complexT> res(hforbs, JRed, NonRedundantMap, ngrid);
    MatrixXT hess;
    boost::function<T (const VectorXT&, VectorXT&, bool)> totalGrad
        = boost::bind(&GetResidual<T, complexT>::getResidue, &res, _1, _2, hess, true, true, false, _3);
    variables.block(0,0,nJastrowVars,1) = JnonRed;
    variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;
    VectorXd residual = variables;

    boost::function<T (const VectorXT&, VectorXT&, bool)> OrbitalGrad
        = boost::bind(&GetResidual<T, complexT>::getOrbitalResidue, &res, boost::ref(JnonRed), _1, _2, _3);
    boost::function<T (const VectorXT&, VectorXT&, bool)> JastrowGrad
        = boost::bind(&GetResidual<T,complexT>::getJastrowResidue, &res, _1, boost::ref(braVars), _2, _3);

    double E = totalGrad(variables, residual, true);
    if (commrank == 0) cout << "INITIAL E: "<<E<<endl;
    cout << getTime()-startofCalc<<endl;
    cout << residual.block(0,0,norbs, 1)<<endl<<endl;
    cout << norbs<<"  "<<JnonRed.size()<<endl;

    GetResidualGJ<T, complexT> resGJ(ngrid); VectorXT Lambda(norbs); Lambda.setZero();
    T Egj = resGJ.getLagrangian(hforbs.block(0,0,2*norbs, nelec), JnonRed, Lambda);
    cout << Egj<<endl;
    exit(0);
    
    if (!(schd.restart || schd.fullrestart)) {
      auto ograd = [&] (VectorXd& vars, VectorXd& res, double& E0, double& stddev, double& rt) {
        E0 = OrbitalGrad(vars, res, true);
        stddev = 0; rt = 0;
        return 1.0;
      };
      auto jgrad = [&] (VectorXd& vars, VectorXd& res, double& E0, double& stddev, double& rt) {
        E0 = JastrowGrad(vars, res, true);
        stddev = 0; rt = 0;
        return 1.0;
      };
      
      AMSGrad optimizerOrb(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      AMSGrad optimizerJas(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      optimizerOrb.optimize(braVars, ograd, schd.restart);
      //optimizerJas.optimize(JnonRed, jgrad, schd.restart);
      variables.block(0,0,nJastrowVars,1) = JnonRed;
      variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;
      saveWave(variables, w, NonRedundantMap, JRed);
    }


    SGDwithDIIS(variables, totalGrad, schd.maxIter, 5.e-4, true);      
    NewtonMethod(variables, totalGrad, schd.maxIter, 1.e-6);

    for (int i=0; i<schd.maxMacroIter; i++)
    {
      if (commrank == 0) cout << "orbital opt"<<endl;      
      SGDwithDIIS(braVars, OrbitalGrad, schd.maxIter, 1.e-4, true);

      if (commrank == 0) cout << "jastrow opt"<<endl;
      SGDwithDIIS(JnonRed, JastrowGrad, schd.maxIter, 1.e-6, true);


      variables.block(0,0,nJastrowVars,1) = JnonRed;
      variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;
      double E = totalGrad(variables, residual, true);
      if (commrank == 0) cout << "----- ITER "<<i<<" -----  "<<E<<"  "<<residual.norm()<<endl;
      saveWave(variables, w, NonRedundantMap, JRed);
    }


    //Solve the non-linear equations using SGD+DIIS and Newton
    if (true)
    {
      //SGDwithDIIS(variables, totalGrad, 100, 5.e-4);      
      NewtonMethod(variables, totalGrad, schd.maxIter, 1.e-6);
    }
    saveWave(variables, w, NonRedundantMap, JnonRed);

    /*
    //Using stan to get gradient of variance
    {
      MatrixXT hess;
      boost::function<T (const VectorXT&)> totalGrad
          = boost::bind(&GetResidual<T, complexT>::getVariance, &res, _1, true, true, false);
      variables.block(0,0,nJastrowVars,1) = JnonRed;
      variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;

      T variance = totalGrad(variables);
      auto getGrad = [&totalGrad](VectorXd& variables, VectorXd& grad, double& E0, double& stddev, double &rt) mutable -> double
      {
        variance.grad();
        //for (int i=0; i<variables.size(); i++)
        //cout << variables[i].adj()<<endl;

      }
    }


    //variance minimization with Newton method
    {
      boost::function<T (const VectorXT&, VectorXT&, bool)> totalGrad
          = boost::bind(&GetResidual<T, complexT>::getResidue, &res, _1, _2, hess,
                        true, true, false, _3);
      
      
      NewtonMethodMinimize(variables, totalGrad, schd.maxIter, 1.e-6);
      
    }
    
    //variance minmization with amsgrad
    {
      auto getGrad = [&res,&hess](VectorXd& variables, VectorXd& grad, double& E0, double& stddev, double &rt) mutable -> double
      {
        grad = variables; grad.setZero();
        double eps = 1.e-5;
        E0 = res.getResidue(variables, grad, hess, true, true, false, true);
        double var = grad.stableNorm();
        for (int i=commrank; i<grad.rows(); i+=commsize) {
          double temp = variables[i];
          double h = eps*abs(temp);
          if (h <1.e-14)
            h = eps;
          
          variables[i] += h;
          grad[i] = (res.getVariance(variables, true, true, false)-var)/h;
          variables[i] = temp;
        }
        int jsize = grad.rows();
        MPI_Allreduce(MPI_IN_PLACE, &grad(0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        stddev = var;
        rt = 0.0;
        return 1.0;
      };
      AMSGrad amsgrad;
      amsgrad.optimize(variables, getGrad, false);
    }
    */
    /*
    for (int i=0; i<8; i++)
    {
      boost::function<T (const VectorXT&, VectorXT&)> OrbitalGrad
          = boost::bind(&GetResidual<T, complexT>::getOrbitalResidue, &res, boost::ref(JnonRed), _1, _2);
      //NewtonMethod(braVars, OrbitalGrad, schd.maxIter, 5.e-4, true);
      SGDwithDIIS(braVars, OrbitalGrad, 100, schd.maxIter, true);

      //if (commrank == 0) cout << "Jastrow optimization "<<endl;
      boost::function<T (const VectorXT&, VectorXT&)> JastrowGrad
          = boost::bind(&GetResidual<T,complexT>::getJastrowResidue, &res, _1, boost::ref(braVars), _2);
      //NewtonMethod(JnonRed, JastrowGrad, schd.maxIter, 5.e-4, true);
      SGDwithDIIS(JnonRed, JastrowGrad, schd.maxIter, 1.e-6);
    }
    
    DIIS<T> diis(8, variables.size());
    VectorXT Jcopy = JnonRed;
    VectorXT braCopy = braVars;
    VectorXT residue = variables;
    for (int i=0; i<8; i++)
    {
      //if (commrank == 0) cout <<endl<< "Orbital optimization "<<endl;
      boost::function<T (const VectorXT&, VectorXT&)> OrbitalGrad
          = boost::bind(&GetResidual<T, complexT>::getOrbitalResidue, &res, boost::ref(Jcopy), _1, _2);
      //NewtonMethod(braVars, OrbitalGrad, schd.maxIter, 5.e-4, false);
      SGDwithDIIS(braVars, OrbitalGrad, 100, 1.e-6, true);

      //if (commrank == 0) cout << "Jastrow optimization "<<endl;
      boost::function<T (const VectorXT&, VectorXT&)> JastrowGrad
          = boost::bind(&GetResidual<T,complexT>::getJastrowResidue, &res, _1, boost::ref(braCopy), _2);
      //NewtonMethod(JnonRed, JastrowGrad, schd.maxIter, 5.e-4, true);
      SGDwithDIIS(JnonRed, JastrowGrad, schd.maxIter, 1.e-6);

      //fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);
      //MatrixXcT&& bra = fillWfnOrbs(w.getRef().HforbsA, braVars);
      //w.getRef().HforbsA.block(0,0,2*norbs, nelec) = bra;
      //w.writeWave();

      residue.block(0,0,nJastrowVars,1) = JnonRed - Jcopy;
      residue.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars - braCopy;

      variables.block(0,0,nJastrowVars,1) = JnonRed;
      variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;

      diis.update(variables, residue);
      T E = totalGrad(variables, residue);
      if (commrank == 0) cout << i<<"  "<<E<<"  "<<residue.norm()<<endl;
      
      JnonRed= variables.block(0,0,nJastrowVars,1) ;
      braVars = variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) ;

      Jcopy = JnonRed;
      braCopy = braVars;
            
    }
    */
    /*
    //concerted optimization
    if (true)
    {      
      SGDwithDIIS(variables, totalGrad, schd.maxIter, 5.e-4);
      NewtonMethod(variables, totalGrad, schd.maxIter, 5.e-4);
      //saveWave(variables, w);
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);

    //Levenberg Helper
    {
      boost::function<T (const VectorXT&, VectorXT&)> totalGrad
          = boost::bind(&GetResidual<T,complexT>::getResidue, &res, _1, _2, hess, true, true, false, false);
      typedef totalGradWrapper<boost::function<T (const VectorXT&, VectorXT&)>> Wrapper;
      
      Wrapper wrapper(totalGrad, variables.rows(), variables.rows());

      {
        VectorXT residuals = variables;
        if (commrank == 0) cout << totalGrad(variables, residuals)<<endl;
      }
      
      Eigen::NumericalDiff<Wrapper> numDiff(wrapper);
      LevenbergMarquardt<Eigen::NumericalDiff<Wrapper>> LM(numDiff);
      int ret = LM.minimize(variables);

      VectorXT residuals = variables;
      if (commrank == 0) {
        std::cout << ret <<endl;
        cout<< LM.iter<<endl;
        std::cout << "Energy at minimum: " << totalGrad(variables, residuals) << std::endl;
        std::cout << "Residue at minimum: " << residuals.norm() << std::endl;
        //saveWave(variables, w);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    exit(0);
    /*
    //first optimize the jastrow
    for (int i=0; i<8; i++)
    {

      if (commrank == 0) cout <<endl<< "Orbital optimization "<<endl;
      boost::function<T (const VectorXT&, VectorXT&)> OrbitalGrad
          = boost::bind(&GetResidual<T,complexT>::getOrbitalResidue, &res, boost::ref(JA), _1, _2);
      SGDwithDIIS(braVars, OrbitalGrad, 50, 1.e-6);

      if (commrank == 0) cout << "Jastrow optimization "<<endl;
      boost::function<T (const VectorXT&, VectorXT&)> JastrowGrad
          = boost::bind(&GetResidual<T,complexT>::getJastrowResidue, &res, _1, boost::ref(braVars), _2);
      SGDwithDIIS(JA, JastrowGrad, 50, 1.e-6);

      
      fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);
      MatrixXcT&& bra = fillWfnOrbs(w.getRef().HforbsA, braVars);
      w.getRef().HforbsA.block(0,0,2*norbs, nelec) = bra;
      w.writeWave();
    }
    */
    return 1.0;
  }


};



/*
    /*
    {
      //using Walker = walker<Jastrow, Slater>;
      VectorXd JAresidue(2*norbs*(2*norbs+1)/2); JAresidue.setZero();
      VectorXd rdm(2*norbs*(2*norbs+1)/2); rdm.setZero();
      MatrixXT Onerdm(2*norbs,2*norbs); Onerdm.setZero();

      Walker<Jastrow, Slater> walk;
      std::vector<Determinant> allDets;
      generateAllDeterminants(allDets, norbs, nalpha, nbeta);
      
      double Energytrans = 0.0, Energyvar = 0.0;
      double denominatortrans = 0.0, denominatorvar = 0.0;
      workingArray work;
      
      for (int i = commrank; i < allDets.size(); i += commsize)
      {
        double ovlp, Eloc;
        w.initWalker(walk, allDets[i]);
        w.HamAndOvlp(walk, ovlp, Eloc, work, false);
        double corrovlp = w.getCorr().Overlap(walk.d),
            slaterovlp = walk.getDetOverlap(w.getRef());
        //cout << allDets[i]<<"  "<<ovlp<<endl;

        Energytrans += slaterovlp/corrovlp * Eloc*ovlp;
        denominatortrans += slaterovlp*slaterovlp;

        Energyvar +=  Eloc*ovlp*ovlp;
        denominatorvar += ovlp*ovlp;

        for (int orb1 = 0; orb1 < 2*norbs; orb1++)
        for (int orb2 = 0; orb2 <= orb1; orb2++)
        {
          if ( allDets[i].getocc(orb1) && allDets[i].getocc(orb2)) {
            int I = (orb1/2) + (orb1%2)*norbs;
            int J = (orb2/2) + (orb2%2)*norbs;
            int K = max(I,J), L = min(I,J);
            JAresidue(K * (K+1)/2 + L) += slaterovlp/corrovlp * Eloc * ovlp;
            rdm(K * (K+1)/2 + L) += slaterovlp*slaterovlp;
          }          
        }

      }

#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energytrans), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(denominatortrans), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Energyvar), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(denominatorvar), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      
      if (commrank == 0) {
        cout << JAresidue(0)<<"  "<<Energytrans/denominatortrans<<"  "<<rdm(0)<<endl;
        cout << Energytrans<<"  "<<denominatortrans <<endl;
        cout << Energyvar<<"  "<<denominatorvar <<endl;
        cout <<"trans energy "<< Energytrans/denominatortrans <<endl;
        cout <<"var energy "<< Energyvar/denominatorvar <<endl;
        JAresidue -= (Energytrans/denominatortrans) * rdm;
        JAresidue /= denominatortrans;
        //cout << "residue"<<endl;
        //cout <<endl<< JAresidue <<endl<<endl;
        cout << "residue norm: "<<JAresidue.norm() <<endl;
      }

      VectorXd residueJA(JA);
      if (commrank == 0) cout << res.getJastrowResidue(JA, braVars, residueJA)<<"  "<<residueJA.norm()<<endl;
      
      double diffnorm = 0.0;
      for (int i=0; i<2*norbs; i++) {
        for (int j=0; j<=i; j++) {
          cout <<i<<"  "<<j<<"  "<< JAresidue(i*(i+1)/2 + j) <<"  "<<residueJA(i * (i+1)/2 + j )<<endl; ;
          diffnorm += pow(JAresidue(i*(i+1)/2 + j) - residueJA((i) * (i+1)/2 + j), 2 );
        }
      }
      cout << diffnorm<<endl;
      exit(0);
    }
    

    //if (commrank == 0) cout <<endl<< "Combined optimization "<<endl;
    //boost::function<double (const VectorXd&, VectorXd&)> totalGrad
    //  = boost::bind(&GetResidual::getResidue, &res, _1, _2, true, true);

    /*
    if (commrank == 0) cout <<"variables"<<endl<< variables <<endl<<endl;
    VectorXd residue=variables; residue.setZero();
    totalGrad(variables, residue);
    if (commrank == 0) cout <<"residue "<<endl<< residue.block(0,0,nJastrowVars, 1)<<endl<<endl;
    if (commrank == 0) cout << residue.block(nJastrowVars,0,braVars.size(), 1)<<endl<<endl;
    if (commrank == 0) cout << residue.maxCoeff()<<"  "<<residue.minCoeff()<<endl;
    if (commrank == 0) cout << "norm "<<residue.norm()<<endl;
    exit(0);

    
    //SGDwithDIIS(variables, totalGrad, 150, 1.e-6);
    //JA = variables.block(0,0,nJastrowVars,1);
    //braVars = variables.block(nJastrowVars,0,braVars.size(),1);

    //fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);
    //fillWfnOrbs(w.getRef().HforbsA, braVars);
    //w.writeWave();

    /*
    //if (commrank == 0) cout << variables <<endl<<endl;
    VectorXd residue=variables; residue.setZero();
    totalGrad(variables, residue);
    if (commrank == 0) cout << residue.block(0,0,nJastrowVars, 1)<<endl<<endl;
    if (commrank == 0) cout << residue.block(nJastrowVars,0,braVars.size(), 1)<<endl<<endl;
    if (commrank == 0) cout << residue.maxCoeff()<<"  "<<residue.minCoeff()<<endl;
    if (commrank == 0) cout << "norm "<<residue.norm()<<endl;
    exit(0);

    //NewtonMethod(variables, totalGrad);
    

  ADDITIONAL CODE USED IN THE PAST FOR TESTING
   I AM KEEPING IT AROUND IN CASE WE NEED TO USE IT AGAIN.

    /*
    {
      VectorXd JAresidue(2*norbs*(2*norbs+1)/2); JAresidue.setZero();
      VectorXd rdm(2*norbs*(2*norbs+1)/2); rdm.setZero();
      MatrixXd Onerdm(2*norbs,2*norbs); Onerdm.setZero();

      Walker<Jastrow, Slater> walk;
      std::vector<Determinant> allDets;
      generateAllDeterminants(allDets, norbs, nalpha, nbeta);
      //vector<vector<int> > alldetvec;
      //comb(2*norbs, nalpha+nbeta, alldetvec);
      
      double Energytrans = 0.0, Energyvar = 0.0;
      double denominatortrans = 0.0, denominatorvar = 0.0;
      workingArray work;
      
      for (int i = commrank; i < allDets.size(); i += commsize)
      {
        double ovlp, Eloc;
        w.initWalker(walk, allDets[i]);
        w.HamAndOvlp(walk, ovlp, Eloc, work, false);
        double corrovlp = w.getCorr().Overlap(walk.d),
            slaterovlp = walk.getDetOverlap(w.getRef());
        //cout << allDets[i]<<"  "<<ovlp<<endl;

        Energytrans += slaterovlp/corrovlp * Eloc*ovlp;
        denominatortrans += slaterovlp*slaterovlp;

        Energyvar +=  Eloc*ovlp*ovlp;
        denominatorvar += ovlp*ovlp;

        for (int orb1 = 0; orb1 < 2*norbs; orb1++)
        for (int orb2 = 0; orb2 <= orb1; orb2++)
        {
          if (allDets[i].getocc(orb1) && allDets[i].getocc(orb2)) {
            int I = (orb1/2) + (orb1%2)*norbs;
            int J = (orb2/2) + (orb2%2)*norbs;
            int K = max(I,J), L = min(I,J);
            JAresidue(K * (K+1)/2 + L) += slaterovlp/corrovlp * Eloc * ovlp;
            rdm(K * (K+1)/2 + L) += slaterovlp*slaterovlp;
          }          
        }

      }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energytrans), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(denominatortrans), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Energyvar), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(denominatorvar), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      
      if (commrank == 0) {
        cout << Energytrans<<"  "<<denominatortrans <<endl;
        cout << Energyvar<<"  "<<denominatorvar <<endl;
        cout <<"trans energy "<< Energytrans/denominatortrans <<endl;
        cout <<"var energy "<< Energyvar/denominatorvar <<endl;
        JAresidue -= (Energytrans/denominatortrans) * rdm;
        JAresidue /= denominatortrans;
        //cout << "residue"<<endl;
        //cout << JAresidue <<endl;
        cout << "residue norm: "<<JAresidue.norm() <<endl;
      }
    }
/////////////////
    
   

    /*
    {

      cout << "Jastrow optimization "<<endl;
      VectorXd braVars(2* (2*norbs-nelec) * nelec); braVars.setZero();
      boost::function<int (const VectorXd&, VectorXd&)> JastrowGrad
          = boost::bind(&GetResidual::getJastrowResidue, &res, _1, boost::ref(braVars), _2);
      NewtonMethod(JA, JastrowGrad);

      HybridNonLinearSolver<boost::function<int (const VectorXd&, VectorXd&)>> solver(JastrowGrad);
      solver.solveNumericalDiffInit(variables);
      int info = solver.solveNumericalDiff(JA);
      cout << info <<endl;
      VectorXd Jastrowres = JA;
      cout << JastrowGrad(JA, Jastrowres)<<endl;
      cout << Jastrowres.norm()<<endl;
    }
///////////////////////



    /*
    variables.block(0,0,nJastrowVars, 1) = JA;
    variables.block(nJastrowVars,0,braVars.size(),1) = braVars;

    cout <<endl<< "Combined optimization "<<endl;
    boost::function<double (const VectorXd&, VectorXd&)> totalGrad
        = boost::bind(&GetResidual::getResidue, &res, _1, _2, true, true);


    //optimizeJastrowParams(variables, res);
    NewtonMethod(variables, totalGrad);
    //SGDwithDIIS(variables, totalGrad);
    */

    
    //optimizeOrbitalParams(variables, totalGrad, residual);
    
    //HybridNonLinearSolver<boost::function<int (const VectorXd&, VectorXd&)>> solver(totalGrad);
    //solver.solveNumericalDiffInit(variables);
    //int info = solver.solveNumericalDiff(variables);
    //totalGrad(variables, residue);
    //double norm = residue.norm();
    //std::cout << format("%14.8f   %14.6f \n") %(residual.Energy()) %(norm);


    /*
    boost::function<int (const VectorXd&, VectorXd&)> fJastrow
        = boost::bind(&Residuals::getJastrowResidue, &residual, _1, _2);
    boost::function<int (const VectorXd&, VectorXd&)> forb
        = boost::bind(&getOrbGradient, boost::ref(residual), _1, _2);

    //double norm = 10.;

    //CERES SOLVER
    /*
    {
      char ** argv;
      google::InitGoogleLogging(argv[0]);
      Problem problem;
      //boost::function<bool (const double* const, double*)> costfun
      //  = boost::bind(&NumericCeresCostFunc, boost::ref(residual), _1, _2);
      NumericCeresCostFunc costfun(residual);
      
      CostFunction* cost_function =
          new DynamicNumericDiffCostFunction<NumericCeresCostFunc, ceres::CENTRAL>(costfun);
      problem.AddResidualBlock(cost_function, NULL, &JA(0));
      
      Solver::Options options;
      options.minimizer_progress_to_stdout = true;
      Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.BriefReport() << "\n";
    }
    */
    /*
    {
      cout << residual.Energy() <<endl;
      ResidualGradient::Residuals residue(norbs, nalpha, nbeta, bra, ngrid);
      double error;
      VectorXd grad = JA;
      for (int i=0; i<300; i++) {
        stan::math::gradient(residue, JA, error, grad);
        JA -= 0.002* grad;
        cout <<"RESIDUE "<< error <<"  "<<grad.norm()<<endl;
      }
      cout << residual.Energy() <<endl;
    }
    */
    //auto Jresidue = JA;
    //fJastrow(JA, Jresidue);
    /*
    HybridNonLinearSolver<boost::function<int (const VectorXd&, VectorXd&)>> solver(fJastrow);
    solver.solveNumericalDiffInit(JA);
    int info = solver.solveNumericalDiff(JA);
    fJastrow(JA, Jresidue);
    norm = Jresidue.norm();
    std::cout << format("%14.8f   %14.6f \n") %(residual.Energy()) %(norm);
    cout << info <<endl;
    //optimizeOrbitalParams(braReal, forb, residual);

    int iter = 0;    
    while(norm > 1.e-6 && iter <10) {
      //cout << JA <<endl;
      int info = solver.solveNumericalDiffOneStep(JA);
      residual.Jastrow = JA;
      fJastrow(JA, Jresidue);
      norm = Jresidue.norm();
      cout <<"info: "<< info <<endl;
      std::cout << format("%14.8f   %14.6f \n") %(residual.E0) %(norm);
      iter++;
    }
    cout << residual.Energy() <<endl;
    optimizeJastrowParams(JA, fJastrow, residual);
    optimizeOrbitalParams(braReal, forb, residual);
    exit(0);
    */
    //optimizeJastrowParams(JA, fJastrow, residual);
    /*
    for (int i=0; i<20; i++) {
      optimizeJastrowParams(JA, fJastrow, residual);
      optimizeOrbitalParams(braReal, forb, residual);
    }
    */
    //optimizeOrbitalParams(braReal, forb, residual);
    //optimizeJastrowParams(JA, fJastrow, residual);
    //optimizeOrbitalParams(braReal, forb, residual);


    /*
    cout << "Optimizing orbitals"<<endl;
    VectorXd braReal(2*2*norbs * (nalpha+nbeta));
    VectorXd braResidue(2*2*norbs * (nalpha+nbeta));
    for (int i=0; i<2*norbs; i++) 
      for (int j=0; j<bra.cols(); j++) {
        braReal(2*i*bra.cols()+j  ) = bra(i,j).real();
        braReal(2*i*bra.cols()+j+1) = bra(i,j).imag();
      }
    //cout <<endl<<endl<< braReal<<endl<<endl;
    double Energy;
    optimizeParams(braReal, forb, residual);

    cout << "Optimizing Jastrows"<<endl;
    optimizeParams(JA, fJastrow, residual);
    
    cout << residual.bra<<endl<<endl;
    cout << residual.ket[0]<<endl<<endl;
    cout << residual.ket[1]<<endl<<endl;

    //cout <<endl<<endl<< braReal<<endl<<endl;
    
    cout << residual.Energy() <<endl;
    optimizeParams(braReal, forb, residual);

    cout << "Optimizing Jastrows"<<endl;
    optimizeParams(JA, fJastrow, residual);
    exit(0);
    */
