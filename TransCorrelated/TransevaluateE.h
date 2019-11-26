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
#include "CorrelatedWavefunction.h"

#include "LevenbergHelper.h"
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>
#include "stan/math.hpp"
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
void saveWave(VectorXd& variables, Wfn& w) {
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  int nelec = nalpha + nbeta;
  
  int nJastrowVars = 2*norbs*(2*norbs+1)/2;
  int nOrbitalVars = (2*norbs-nelec)*(nelec);
  VectorXd JA = variables.block(0,0,nJastrowVars,1);
  VectorXd braVars = variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1);
  fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);
  MatrixXcd&& bra = fillWfnOrbs(w.getRef().HforbsA, braVars);
  w.getRef().HforbsA.block(0,0,2*norbs, nelec) = bra;
  w.writeWave();

  variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1).setZero();
}



template <typename Wfn>
class getTranscorrelationWrapper
{
 public:
  Wfn &w;
  getTranscorrelationWrapper(Wfn &pwr) : w(pwr)
  {};

  double optimizeWavefunction()
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    int nelec = nalpha + nbeta;
    
    int nJastrowVars = 2*norbs*(2*norbs+1)/2;
    int nOrbitalVars = (2*norbs-nelec)*(nelec);
    VectorXd variables(nJastrowVars + 2*nOrbitalVars);

    VectorXd JA(nJastrowVars);
    fillJastrowfromWfn(w.getCorr().SpinCorrelator, JA);
    VectorXd braVars(2* nOrbitalVars); braVars.setZero();
    int ngrid = 4; //FOR THE SZ PROJECTOR

    GetResidual res(w.getRef().HforbsA, ngrid);//calculates orbital, Jastrow Residue

    //first optimize orbitals
    /*
    {
      if (commrank == 0) cout <<endl<< "Orbital optimization "<<endl;
      boost::function<double (const VectorXd&, VectorXd&)> OrbitalGrad
          = boost::bind(&GetResidual::getOrbitalResidue, &res, boost::ref(JA), _1, _2);
      SGDwithDIIS(braVars, OrbitalGrad, 100, 1.e-6);
    }
    */
    
    MatrixXd hess;
    boost::function<double (const VectorXd&, VectorXd&)> totalGrad
        = boost::bind(&GetResidual::getResidue, &res, _1, _2, hess, true, true, false);
    variables.block(0,0,nJastrowVars,1) = JA;
    variables.block(nJastrowVars, 0, 2*nOrbitalVars, 1) = braVars;

    {
      VectorXd residuals = variables;
      double E = totalGrad(variables, residuals);
      if (commrank == 0) cout << E<<endl;
      MPI_Barrier(MPI_COMM_WORLD);
      exit(0);
    }

    //concerted optimization
    if (true)
    {      
      SGDwithDIIS(variables, totalGrad, schd.maxIter, 1.e-6);
      saveWave(variables, w);
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);
    
    //Levenberg Helper
    {
      typedef totalGradWrapper<boost::function<double (const VectorXd&, VectorXd&)>> Wrapper;
      
      Wrapper wrapper(totalGrad, variables.rows(), variables.rows());

      {
        VectorXd residuals = variables;
        if (commrank == 0) cout << totalGrad(variables, residuals)<<endl;
      }
      
      Eigen::NumericalDiff<Wrapper> numDiff(wrapper);
      LevenbergMarquardt<Eigen::NumericalDiff<Wrapper>> LM(numDiff);
      int ret = LM.minimize(variables);

      VectorXd residuals = variables;
      if (commrank == 0) {
        std::cout << ret <<endl;
        cout<< LM.iter<<endl;
        std::cout << "Energy at minimum: " << totalGrad(variables, residuals) << std::endl;
        std::cout << "Residue at minimum: " << residuals.norm() << std::endl;
        saveWave(variables, w);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    exit(0);
    //first optimize the jastrow
    for (int i=0; i<8; i++)
    {

      if (commrank == 0) cout <<endl<< "Orbital optimization "<<endl;
      boost::function<double (const VectorXd&, VectorXd&)> OrbitalGrad
          = boost::bind(&GetResidual::getOrbitalResidue, &res, boost::ref(JA), _1, _2);
      SGDwithDIIS(braVars, OrbitalGrad, 50, 1.e-6);

      if (commrank == 0) cout << "Jastrow optimization "<<endl;
      boost::function<double (const VectorXd&, VectorXd&)> JastrowGrad
          = boost::bind(&GetResidual::getJastrowResidue, &res, _1, boost::ref(braVars), _2);
      SGDwithDIIS(JA, JastrowGrad, 50, 1.e-6);

      
      fillWfnfromJastrow(JA, w.getCorr().SpinCorrelator);
      MatrixXcd&& bra = fillWfnOrbs(w.getRef().HforbsA, braVars);
      w.getRef().HforbsA.block(0,0,2*norbs, nelec) = bra;
      w.writeWave();
    }
    return 1.0;
  }


};



/*
    /*
    {
      //using Walker = walker<Jastrow, Slater>;
      VectorXd JAresidue(2*norbs*(2*norbs+1)/2); JAresidue.setZero();
      VectorXd rdm(2*norbs*(2*norbs+1)/2); rdm.setZero();
      MatrixXd Onerdm(2*norbs,2*norbs); Onerdm.setZero();

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
