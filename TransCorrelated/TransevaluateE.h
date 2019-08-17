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


int getOrbGradient(Residuals& residue, const VectorXd& braReal, VectorXd& braResidue) {
  double Energy;
  //stan::math::gradient(residue, braReal, Energy, braResidue);
  //residue.E0 = Energy;

  //VectorXd braResidue2 = braResidue;
  Energy = residue.getOrbitalResidue(braReal, braResidue);
  residue.E0 = Energy;
  
  /*
  cout << Energy;  
  cout << "braresidue "<<endl;
  //cout << braResidue <<endl<<endl;
  //cout << "norm "<<braResidue.norm() <<endl<<endl;
  VectorXd braResidue2 = braResidue;
  //VectorXd braReal2 = braReal;
  //braReal2(0) += 1.e-3;
  double E2 = residue.getOrbitalResidue(braReal, braResidue2);
  //cout <<"grad "<< (E2 - E1)/1.e-3<<endl;
  //cout << "E1 E2 "<<E2<<"  "<<E1<<endl;
  //cout <<"Energy "<< residue.getOrbitalResidue(braReal, braReal2, braResidue2)<<endl;
  cout << "residue "<<endl;
  cout << braResidue2<<endl<<endl;
  //exit(0);
  */
  //calculate the overlap <bra|ket[g]> * coeff[g]
  /*
  Matrix<stan::math::var, Dynamic, 1> braVarReal(braReal.size());
  for (int i=0; i<braReal.size(); i++)
    braVarReal(i) = braReal(i);


  stan::math::var detovlp = residue.getOvlp(braVarReal);
  double denominator = detovlp.val();
  detovlp.grad();
  for (int i=0; i<braReal.size(); i++)
    braResidue(i) -= Energy * braVarReal(i).adj()/denominator;

  cout << braResidue <<endl<<endl;
  cout <<braResidue2<<endl<<endl;
  //cout << "residue "<<endl<<braResidue<<endl;
  cout << "norm"<< (braResidue-braResidue2).norm()<<endl;
  exit(0);
  */
  return 0;
}

double SingleGradient(const VectorXd& variables, VectorXd& residuals) {
  double E = getResidual(variables, residuals);
  //double E = getGradient(variables, residuals);
  return E;
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
    */
    
    MatrixXcd bra = w.getRef().HforbsA.block(0,0,2*norbs, nalpha+nbeta);
    

    //The place holder for jastrow variables and jastrow residues
    VectorXd JAresidue(2*norbs*(2*norbs+1)/2); JAresidue.setZero();
    VectorXd JA       (2*norbs*(2*norbs+1)/2);

    //the place holoder for complex orbitals and complex orbital residues
    VectorXd orbVars(2* 2*norbs * (nalpha+nbeta));
    VectorXd orbResidue(2* 2*norbs * (nalpha+nbeta)) ; orbResidue.setZero();

    
    //grid to project out the Sz quantum number
    int ngrid = 4;
    //cin >> ngrid;
    
    fillJastrowfromWfn(w.getCorr().SpinCorrelator, JA);

    //JA += 0.1*VectorXd::Random(JA.size());
    VectorXd braReal(2*2*norbs * (nalpha+nbeta));
    for (int i=0; i<2*norbs; i++) 
      for (int j=0; j<bra.cols(); j++) {
        braReal(2*(i*bra.cols()+j)  ) = bra(i,j).real();
        braReal(2*(i*bra.cols()+j)+1) = bra(i,j).imag();
      }

    Residuals residual(norbs, nalpha, nbeta, JA, bra, ngrid);


    int nJastrowVars = 2*norbs*(2*norbs+1)/2;
    int nOrbitalVars = 2*norbs*(nalpha+nbeta);
    VectorXd variables(nJastrowVars + 2*nOrbitalVars);
    variables.block(0,0,nJastrowVars, 1) = JA;
    variables.block(nJastrowVars,0,2*nOrbitalVars,1) = braReal;
    auto residue = variables;

    boost::function<double (const VectorXd&, VectorXd&)> totalGrad
        = boost::bind(&SingleGradient, _1, _2);

    optimizeJastrowParams(variables, totalGrad, residual);

    //optimizeOrbitalParams(variables, totalGrad, residual);
    
    //HybridNonLinearSolver<boost::function<int (const VectorXd&, VectorXd&)>> solver(totalGrad);
    //solver.solveNumericalDiffInit(variables);
    //int info = solver.solveNumericalDiff(variables);
    //totalGrad(variables, residue);
    //double norm = residue.norm();
    //std::cout << format("%14.8f   %14.6f \n") %(residual.Energy()) %(norm);
    exit(0);

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





