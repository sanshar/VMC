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

#include <Eigen/Dense>
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include "DirectJacobian.h"
#include "diis.h"
#include <boost/format.hpp>
#include <unsupported/Eigen/IterativeSolvers>
#include <Eigen/QR>
#include "ParallelJacobian.h"

using namespace Eigen;
using namespace std;
using namespace boost;


template<typename Functor, typename T>
void NewtonMethodMinimize(
    VectorXd& params,
    Functor& func,
    int maxIter = 50,
    T targetTol = 1.e-5,
    bool print = true) {

  T Energy, norm;
  int iter = 0;
  VectorXd residue(params.size());
  //DIIS diis(8, params.size());

  ParallelJacobian<T, Functor>
      Jpar(func, params.rows(), params.rows()); //the jacobian for variance
  Eigen::GMRES<ParallelJacobian<T, Functor>, Eigen::IdentityPreconditioner> gmres;
  //Eigen::ConjugateGradient<ParallelJacobian<T, Functor>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner>  gmres;
  gmres.compute(Jpar);
  gmres.set_restart(100);
  gmres.setTolerance(1.e-6);
  gmres.setMaxIterations(1000);

  Energy = func(params, residue, true);
  norm = residue.norm();
  
  if (commrank == 0) std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
      %(residue.norm()) %(getTime()-startofCalc);

  //various step sizes
  VectorXd testRes = residue, testParams = params; 
  vector<T> testSteps = {0.1, 0.5, 1}; 
  VectorXd Jtb = residue;
  
  while(norm > targetTol && iter < maxIter) {      
    Jpar.PopulateJacobian(params);
    Jtb.setZero();
    Jpar.multiplyWithTranspose(residue, Jtb);
    
    Jtb *=-1.;
    Matrix<T, Dynamic, 1> x = gmres.solve(Jtb);

    //Matrix<T, Dynamic, 1> error = Jpar*x - Jtb;
    //if (commrank == 0) cout << error.stableNorm()<<"  "<<error.stableNorm()/residue.stableNorm()<<endl;

    T testStep = 1.0, testNorm = 1.e20;
    for (int i=0; i<testSteps.size(); i++)
    {
      testParams = params + testSteps[i]*x;
      func(testParams, testRes, true);
      T testResNorm = testRes.norm();
      if (testResNorm < testNorm) {
        testNorm = testResNorm;
        testStep = testSteps[i];
      }
    }
    
    params += testStep*x;
    //diis.update(params, residue);
    iter++;

    
    Energy = func(params, residue, true);
    norm = residue.norm();
    
    if (commrank == 0 && print) std::cout << format("%5i   %14.8f   %14.6f %6.4f %6.4e %5i %6.6f \n") %(iter) %(Energy) %(residue.norm()) %(testStep) %(gmres.error()) %(gmres.iterations()) %(getTime()-startofCalc);
    //exit(0);
  }
}


template<typename Functor, typename T>
void NewtonMethod(
    Matrix<T, Dynamic, 1>& params,
    Functor& func,
    int maxIter = 50,
    T targetTol = 1.e-5,
    bool print = true) {

  //cout << "newton method**"<<endl;

  T Energy, norm;
  int iter = 0;
  Matrix<T, Dynamic, 1> residue(params.size());
  DIIS<T> diis(8, params.size());

  DirectJacobian<T, Functor> J(func, params.rows(), params.rows());
  //Eigen::DGMRES<DirectJacobian<T, Functor>, Eigen::IdentityPreconditioner> gmres;
  //Eigen::BiCGSTAB<DirectJacobian<T, Functor>, Eigen::IdentityPreconditioner> gmres;
  Eigen::GMRES<DirectJacobian<T, Functor>, Eigen::IdentityPreconditioner> gmres;
  gmres.compute(J);
  gmres.set_restart(100);
  gmres.setTolerance(1.e-6);
  gmres.setMaxIterations(1000);
  Energy = func(params, residue, true);
  norm = residue.norm();
  
  if (commrank == 0 && print ) std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
      %(residue.norm()) %(getTime()-startofCalc);

  //various step sizes
  Matrix<T, Dynamic, 1> testRes = residue, testParams = params; 
  vector<T> testSteps = {0.001, 0.01, 0.1, 0.5, 1}; 

  while(norm > targetTol && iter < maxIter) {      
    J.PopulateJacobian(params);
    //J.setFvec(params, residue);

    residue *=-1.;
    Matrix<T, Dynamic, 1> x = gmres.solve(residue);

    Matrix<T, Dynamic, 1> error = J*x - residue;
    //if (commrank == 0) cout << error.stableNorm()<<"  "<<error.stableNorm()/residue.stableNorm()<<endl;

    T testStep = 1.0, testNorm = 1.e20;
    for (int i=0; i<testSteps.size(); i++)
    {
      testParams = params + testSteps[i]*x;
      func(testParams, testRes, true);
      T testResNorm = testRes.norm();
      if (testResNorm < testNorm) {
        testNorm = testResNorm;
        testStep = testSteps[i];
      }
    }
    
    params += testStep*x;
    //diis.update(params, residue);
    iter++;

    
    Energy = func(params, residue, true);
    norm = residue.norm();
    
    if (commrank == 0 && print) std::cout << format("%5i   %14.8f   %14.6f %6.4f %6.4e %5i %6.6f \n") %(iter) %(Energy) %(residue.norm()) %(testStep) %(gmres.error()) %(gmres.iterations()) %(getTime()-startofCalc);
  }
}


template <typename Functor, typename T>
void SGDwithDIIS(
    Matrix<T, Dynamic, 1>& params,
    Functor& func,
    int maxIter = 50,
    double targetTol = 1.e-3,
    bool print = true) {

  T Energy, norm=10.;
  int iter = 0;
  Matrix<T, Dynamic, 1> residue(params.size()); residue.setZero();

  DIIS<T> diis(10, params.size());
  

  Energy = func(params, residue, true);
  norm = residue.stableNorm();

  if (commrank == 0 && print) std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
      %(norm) %(getTime()-startofCalc);

  T eps = 0.01;
  while(norm > targetTol && iter < maxIter) {      
    params -=  eps*residue;
    diis.update(params, residue);

    //if (commrank == 0) cout << diis.diisMatrix<<endl;
    iter++;
    
    Energy = func(params, residue, true);
    norm = residue.stableNorm();

    if (commrank == 0 && print) std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
        %(norm) %(getTime()-startofCalc);
  }
}



template <typename Functor, typename T>
Matrix<T, Dynamic, 1> HybridKrylovDogLegMethod (
    Matrix<T, Dynamic, 1>& params,
    Functor& getRes) {

  Matrix<T, Dynamic, 1> x(params.size()); x.setZero();

  //acts like a matrix that multiplies a vector with approximate Jacobian
  DirectJacobian<T> J(getRes);

  Matrix<T, Dynamic, 1> r(params.size()); r.setZero();  
  getRes(params, r);

  J.setFvec(params, r);

  T beta = r.norm();
      
  //first construct the Krylov vectors
  int maxNumVec = 10;
  vector<Matrix<T, Dynamic, 1>> Vm(1,r);
  MatrixXd Hessen(maxNumVec+1, maxNumVec); Hessen.setZero();

  T targetError = 1.e-3;
  int iter = 1; 
  while (true) {
    int j = Vm.size();

    cout << "before arnoldi"<<endl;

    //arnoldi step
    Matrix<T, Dynamic, 1> Jvj = J* Vm[j-1];
    //cout << Vm[0].size()<<"  "<<Jvj.size()<<endl;

    //gram schmidt orthogonalization
    for (int i=0; i<j; i++) {
      Hessen(i, j-1) = Jvj.transpose() * Vm[i];
      Jvj = Jvj - Hessen(i, j-1) * Vm[i];
    }
    Hessen(j, j-1) = Jvj.norm();
    Jvj /= Hessen(j, j-1);
    Vm.push_back(Jvj);

    
    //cout << "solve least square"<<endl;
    Matrix<T, Dynamic, 1> e(j+1); e.setZero(); e(0) = 1.;
    auto HessenBlock = Hessen.block(0,0, j+1, j);
    Matrix<T, Dynamic, 1> ym = (HessenBlock.transpose()*HessenBlock).inverse()*(HessenBlock.transpose()*e);

    //cout << "calc error"<<endl;
    //cout << Hessen.rows()<<"  "<<Hessen.cols()<<"  "<<ym.size()<<"  "<<e.size()<<endl;
    T error = (HessenBlock*ym - e).norm();
    cout<< HessenBlock <<endl;
    cout << error <<"  "<<iter<<"  "<<maxNumVec<<"  "<<beta<<endl;
    iter++;
    
    if (abs(error) < targetError || iter > maxNumVec) {
      for (int i=0; i<j; i++)
        x += ym(i) * Vm[i];
      return;
    }
  }

}
