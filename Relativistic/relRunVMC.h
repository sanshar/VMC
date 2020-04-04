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
#include "amsgrad.h"
#include "sgd.h"
#include "ftrl.h"
#include "sr.h"
#include "linearMethod.h"
//#include "variance.h"

using functor1r = boost::function<double (VectorXd&, VectorXd&, double&, double&, double&)>;




template<typename Wave, typename Walker>
void runRelVMC(Wave& wave, Walker& walk) {

  if (schd.restart || schd.fullrestart)
    wave.readWave();
  VectorXd vars; wave.getVariables(vars);

  
  if (commrank == 0)
  {
    cout << "Number of Jastrow vars: " << wave.getCorr().getNumVariables() << endl;
    cout << "Number of Reference vars: " << wave.getNumVariables() - wave.getCorr().getNumVariables() << endl;
  }

  relGetGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  functor1r relGetStochasticGradient = boost::bind(&relGetGradientWrapper<Wave, Walker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
  
  if (schd.method == amsgrad || schd.method == amsgrad_sgd) {
      if (schd.stepsizes.empty()) {
        AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
        optimizer.optimize(vars, relGetStochasticGradient, schd.restart);
      }
      else {
        //AMSGrad optimizer(schd.stepsizes, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
        //optimizer.optimize(vars, getStochasticGradient, runCorrelatedSampling, schd.restart); 
        cout << "Not yet implemented for rel" << endl;
      }
  }
  else {
    cout << "No valid optimizer option selected for relativistic calculation" << endl;
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
