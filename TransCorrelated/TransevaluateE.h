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

#include "stan/math.hpp"

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


int getOrbGradient(Residuals& residue, const VectorXd& braReal, VectorXd& braResidue) {
  double Energy;
  stan::math::gradient(residue, braReal, Energy, braResidue);
  residue.E0 = Energy;

  //calculate the overlap <bra|ket[g]> * coeff[g]
  Matrix<stan::math::var, Dynamic, 1> braVarReal(braReal.size());
  for (int i=0; i<braReal.size(); i++)
    braVarReal(i) = braReal(i);

  stan::math::var detovlp = residue.getOvlp(braVarReal);
  double denominator = detovlp.val();
  detovlp.grad();
  for (int i=0; i<braReal.size(); i++)
    braResidue(i) -= Energy * braVarReal(i).adj()/denominator;
  return 0;
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
    
    MatrixXcd bra = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta);
    

    //The place holder for jastrow variables and jastrow residues
    VectorXd JAresidue(2*norbs*(2*norbs+1)/2); JAresidue.setZero();
    VectorXd JA       (2*norbs*(2*norbs+1)/2);

    //the place holoder for complex orbitals and complex orbital residues
    VectorXd orbVars(2* 2*norbs * (nalpha+nbeta));
    VectorXd orbResidue(2* 2*norbs * (nalpha+nbeta)) ; orbResidue.setZero();

    
    //grid to project out the Sz quantum number
    int ngrid = 1;
    cin >> ngrid;
    fillJastrowfromWfn(w.getCorr().SpinCorrelator, JA);

    VectorXd braReal(2*2*norbs * (nalpha+nbeta));
    for (int i=0; i<2*norbs; i++) 
      for (int j=0; j<bra.cols(); j++) {
        braReal(2*(i*bra.cols()+j)  ) = bra(i,j).real();
        braReal(2*(i*bra.cols()+j)+1) = bra(i,j).imag();
      }

    Residuals residual(norbs, nalpha, nbeta, JA, bra, ngrid);

    boost::function<int (const VectorXd&, VectorXd&)> fJastrow
        = boost::bind(&Residuals::getJastrowResidue, &residual, _1, _2);
    boost::function<int (const VectorXd&, VectorXd&)> forb
        = boost::bind(&getOrbGradient, boost::ref(residual), _1, _2);

    //optimizeJastrowParams(JA, fJastrow, residual);
    for (int i=0; i<20; i++) {
      optimizeOrbitalParams(braReal, forb, residual);
      optimizeJastrowParams(JA, fJastrow, residual);
    }
    //optimizeOrbitalParams(braReal, forb, residual);
    //optimizeJastrowParams(JA, fJastrow, residual);
    //optimizeOrbitalParams(braReal, forb, residual);
    exit(0);

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
    */
    exit(0);
    w.writeWave();
    return 1.0;
  }


};





