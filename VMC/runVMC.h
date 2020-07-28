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
#include "evaluateE.h"
#include "amsgrad.h"
#include "sgd.h"
#include "ftrl.h"
#include "sr.h"
#include "linearMethod.h"
#include <functional>
//#include "variance.h"


namespace ph = std::placeholders;

template<typename Wave, typename Walker>
void runVMC(Wave& wave, Walker& walk) {

  if (schd.restart || schd.fullRestart) wave.readWave();
  VectorXd vars; wave.getVariables(vars);

  
  if (commrank == 0)
  {
    //cout << "Number of Jastrow vars: " << wave.getNumJastrowVariables() << endl;
    //cout << "Number of Reference vars: " << wave.getNumReferenceVariables() << endl;
  }
  
  getGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  auto getStochasticGradient = std::bind(&getGradientWrapper<Wave, Walker>::getGradient, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, schd.deterministic);
  auto getStochasticGradientMetric = std::bind(&getGradientWrapper<Wave, Walker>::getMetric, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientMetricRandom = std::bind(&getGradientWrapper<Wave, Walker>::getMetricRandom, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  //auto getStochasticGradientVariance = std::bind(&getGradientWrapper<Wave, Walker>::getVariance, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientHessian = std::bind(&getGradientWrapper<Wave, Walker>::getHessian, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientHessianDirect = std::bind(&getGradientWrapper<Wave, Walker>::getHessianDirect, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, schd.deterministic);
  auto getStochasticGradientHessianRandom = std::bind(&getGradientWrapper<Wave, Walker>::getHessianRandom, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, ph::_8, schd.deterministic);

  CorrSampleWrapper<Wave, Walker> wrap(schd.CorrSampleFrac * schd.stochasticIter);
  auto runCorrelatedSampling = std::bind(&CorrSampleWrapper<Wave, Walker>::run, &wrap, ph::_1, ph::_2);

  if (schd.method == amsgrad || schd.method == amsgrad_sgd) {
      if (schd.stepsizes.empty()) {
        AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
        optimizer.optimize(vars, getStochasticGradient, schd.restart);
      }
      else {
        AMSGrad optimizer(schd.stepsizes, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
        optimizer.optimize(vars, getStochasticGradient, runCorrelatedSampling, schd.restart); 
      }
  }
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
    if (!schd.direct && !schd.random)
    {
      LM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay);
      optimizer.optimize(vars, getStochasticGradientHessian, runCorrelatedSampling, schd.restart); 
      //optimizer.optimize(vars, getStochasticGradientHessian, schd.restart); 
    }
    else if (schd.direct && !schd.random)
    {
      directLM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay, schd.sgdIter, schd.stepsize, schd.decay1, schd.decay2);
      optimizer.optimize(vars, getStochasticGradientHessianDirect, runCorrelatedSampling, schd.restart);
    }
    else if (schd.random)
    {
      randomLM optimizer(schd.maxIter, schd.stepsizes, schd.sgdIter, schd.stepsize, schd.decay1, schd.decay2);
      optimizer.optimize(vars, getStochasticGradientMetricRandom, getStochasticGradientHessianRandom, runCorrelatedSampling, schd.restart);
    }
  }
/*
  else if (schd.method == varLM)
  {
    directVarLM optimizer(schd.maxIter);
    optimizer.optimize(vars, getStochasticGradientVariance, schd.restart);
  } 
*/
  if (schd.printVars && commrank==0) wave.printVariables();
  
}


template<typename Wave, typename Walker>
void runVMCRealSpace(Wave& wave, Walker& walk) {

  if (schd.restart || schd.fullRestart)
    wave.readWave();

  VectorXd vars; wave.getOptVariables(vars);
  getGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  
  if (commrank == 0)
  {
    cout << "Number of Jastrow vars: " << wave.getNumJastrowVariables() << endl;
    cout << "Number of Reference vars: " << wave.getNumRefVariables() << endl;
    cout << "Number of Optimized vars: " << wave.getNumOptVariables() << endl;
  }

  auto getStochasticGradientRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getGradientRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, schd.deterministic);
  auto getStochasticGradientMetricRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getMetricRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientMetricRandomRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getMetricRandomRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientHessianRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getHessianRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, schd.deterministic);
  auto getStochasticGradientHessianDirectRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getHessianDirectRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, schd.deterministic);
  auto getStochasticGradientHessianRandomRealSpace = std::bind(&getGradientWrapper<Wave, Walker>::getHessianRandomRealSpace, &wrapper, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5, ph::_6, ph::_7, ph::_8, schd.deterministic);

  //CorrSampleWrapper<Wave, Walker> wrap(0.15 * schd.stochasticIter);
  CorrSampleWrapper<Wave, Walker> wrap(schd.CorrSampleFrac * schd.stochasticIter);
  auto runCorrelatedSamplingRealSpace = std::bind(&CorrSampleWrapper<Wave, Walker>::runRealSpace, &wrap, ph::_1, ph::_2);

  if (schd.walkerBasis == REALSPACESTO || schd.walkerBasis == REALSPACEGTO) {
    if (schd.method == amsgrad || schd.method == amsgrad_sgd) {
      AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      optimizer.optimize(vars, getStochasticGradientRealSpace, schd.restart);
    }
    else if (schd.method == sgd) {
      SGD optimizer(schd.stepsize, schd.maxIter);
      optimizer.optimize(vars, getStochasticGradientRealSpace, schd.restart);
    }
    else if (schd.method == sr) {
      SR optimizer(schd.maxIter);
      optimizer.optimize(vars, getStochasticGradientMetricRealSpace, schd.restart);
    }
    else if (schd.method == linearmethod) {
      if (schd.direct && !schd.random) {
        directLM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay, schd.sgdIter, schd.stepsize, schd.decay1, schd.decay2);
        optimizer.optimize(vars, getStochasticGradientHessianDirectRealSpace, runCorrelatedSamplingRealSpace, schd.restart);
      }
      else if (schd.random) {
        randomLM optimizer(schd.maxIter, schd.stepsizes, schd.sgdIter, schd.stepsize, schd.decay1, schd.decay2);
        optimizer.optimize(vars, getStochasticGradientMetricRandomRealSpace, getStochasticGradientHessianRandomRealSpace, runCorrelatedSamplingRealSpace, schd.restart);
      }
      else {  
        LM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay);
        optimizer.optimize(vars, getStochasticGradientHessianRealSpace, runCorrelatedSamplingRealSpace, schd.restart);
      }
    }
  }
  if (schd.printVars && commrank==0) wave.printVariables();
}
