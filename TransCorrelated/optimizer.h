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

using namespace Eigen;
using namespace std;


class Residuals;


void optimizeJastrowParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func);

void optimizeOrbitalParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func);


template <typename Functor>
VectorXd HybridKrylovDogLegMethod (
    VectorXd& params,
    Functor& getRes) {

  VectorXd x(params.size()); x.setZero();

  //acts like a matrix that multiplies a vector with approximate Jacobian
  DirectJacobian<double> J(getRes);

  VectorXd r(params.size()); r.setZero();  
  getRes(params, r);

  J.setFvec(params, r);

  double beta = r.norm();
      
  //first construct the Krylov vectors
  int maxNumVec = 10;
  vector<VectorXd> Vm(1,r);
  MatrixXd Hessen(maxNumVec+1, maxNumVec); Hessen.setZero();

  double targetError = 1.e-3;
  int iter = 1; 
  while (true) {
    int j = Vm.size();

    cout << "before arnoldi"<<endl;

    //arnoldi step
    VectorXd Jvj = J* Vm[j-1];
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
    VectorXd e(j+1); e.setZero(); e(0) = 1.;
    auto HessenBlock = Hessen.block(0,0, j+1, j);
    VectorXd ym = (HessenBlock.transpose()*HessenBlock).inverse()*(HessenBlock.transpose()*e);

    //cout << "calc error"<<endl;
    //cout << Hessen.rows()<<"  "<<Hessen.cols()<<"  "<<ym.size()<<"  "<<e.size()<<endl;
    double error = (HessenBlock*ym - e).norm();
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
