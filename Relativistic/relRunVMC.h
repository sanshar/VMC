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

#include <sys/stat.h>
#include "input.h"
#include "relEvaluateE.h"
#include "evaluateE.h"
#include "amsgrad.h"
#include "relAMSGrad.h"
#include "sgd.h"
#include "ftrl.h"
#include "sr.h"
#include "linearMethod.h"
//#include "variance.h"

using functor1r = boost::function<double (VectorXd&, VectorXd&, double&, double&, double&)>;
using functor1rc = boost::function<double (VectorXd&, VectorXd&, std::complex<double>&, double&, double&)>;
using functor2r = boost::function<double (VectorXd&, std::complex<double>&, double&, double&)>;




template<typename Wave, typename Walker>
void runRelVMC(Wave& wave, Walker& walk) {

  if (schd.restart || schd.fullrestart)
    wave.readWave();
  VectorXd vars; wave.getVariables(vars);

  
  if (commrank == 0)
  {
    cout << "Number of Jastrow vars: " << wave.getCorr().getNumVariables() << endl;
    cout << "Number of Reference vars: " << wave.getNumVariables() - wave.getCorr().getNumVariables() << endl;
    if (0==0) {
      cout << "I1SOC" << endl;
      cout << I1SOC(0,0) << endl;
      cout << I1SOC(0,1) << endl;
      cout << I1SOC(1,0) << endl;
      cout << I1SOC(1,1) << endl;
      cout << I1SOC(0,2) << endl;
      cout << I1SOC(2,0) << endl;
      cout << "I2" << endl;
      cout << I2(0,0,0,0) << endl;
      cout << I2(0,1,0,0) << endl;
      cout << I2(0,1,2,3) << endl;
      cout << I2(0,1,2,2) << endl;
      cout << I2(0,2,3,3) << endl;
    }
  }

/*
  if (schd.deterministic == true && schd.ifRelativistic == false){
    if (commrank == 0) cout << "Deterministic calculation" << endl;
    double Energy = 0.0;
    getEnergyDeterministic(wave, walk, Energy);
    cout << "Deterministic energy: " << Energy << endl;
    exit (0);
  }
*/
  // deterministic calculation
  if (schd.onlyEne == true){
    if (commrank == 0) cout << "Relativistic energy calculation" << endl;
    relGetEnergyWrapper<Wave, Walker> wrapper(wave, walk, schd.deterministic);
    functor2r relGetEne = boost::bind(&relGetEnergyWrapper<Wave, Walker>::getEnergy, &wrapper, _1, _2, _3, _4);
    eneOnly eo(schd.deterministic);
    eo.relGetEnergy(vars, relGetEne);
  }
  else if (schd.onlyEne == false && (schd.method == amsgrad || schd.method == amsgrad_sgd)) {
    if (commrank == 0) cout << "Relativistic vmc calculation" << endl;
    relGetGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
    functor1rc relGetGradient = boost::bind(&relGetGradientWrapper<Wave, Walker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
    if (schd.stepsizes.empty()) {
      relAMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      optimizer.optimize(vars, relGetGradient, schd.restart);
    }
    else {
      //AMSGrad optimizer(schd.stepsizes, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      //optimizer.optimize(vars, getStochasticGradient, runCorrelatedSampling, schd.restart); 
      cout << "Not yet implemented for rel" << endl;
    }
  }
  else {
    cout << "No valid option selected for relativistic calculation" << endl;
  }
  
/*
  else if (schd.method == sgd) {
    SGD optimizer(schd.stepsize, schd.momentum, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == ftrl) {
    SGD optimizer(schd.alpha, schd.beta, schd.maxIter);
    optimizer.optimize(vars, getStochasticGradient, schd.restart);
  }
  else if (schd.method == sr) {
    SR optimizer(schd.maxIter);
    optimizer.optimize(vars, getStochasticGradientMetric, schd.restart);
  }
  else if (schd.method == linearmethod)
  {
    if (!schd.direct)
    {
      LM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay);
      optimizer.optimize(vars, getStochasticGradientHessian, runCorrelatedSampling, schd.restart); 
      //optimizer.optimize(vars, getStochasticGradientHessian, schd.restart); 
    }
    else
    {
      directLM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay, schd.sgdIter);
      optimizer.optimize(vars, getStochasticGradientHessianDirect, runCorrelatedSampling, schd.restart);
    }
  }
*/
/*
  else if (schd.method == varLM)
  {
    directVarLM optimizer(schd.maxIter);
    optimizer.optimize(vars, getStochasticGradientVariance, schd.restart);
  } 
*/
  if (schd.printVars && commrank==0) wave.printVariables();
  
}
