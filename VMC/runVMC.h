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
//#include "variance.h"

using CorrSampleFunctor = boost::function<void (std::vector<Eigen::VectorXd>&, std::vector<double>&)>; 
using functor1 = boost::function<double (VectorXd&, VectorXd&, double&, double&, double&)>;
using functor2 = boost::function<void (VectorXd&, VectorXd&, VectorXd&, DirectMetric&, double&, double&, double&)>;
//using functor0 = boost::function<void (VectorXd&, VectorXd&, DirectVarLM &, double&, double&, double&, double&)>;
using functor3 = boost::function<double (VectorXd&, VectorXd&, double&, double&, double&)>;
//using functor4 = boost::function<void (VectorXd&, VectorXd&, VectorXd&, DirectMetric&, double&, double&, double&)>;
using functor5 = boost::function<double (VectorXd&, VectorXd&, MatrixXd&, MatrixXd&, double&, double&, double&)>;
using functor6 = boost::function<double (VectorXd&, VectorXd&, DirectLM&, double&, double&, double&)>;


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
  functor1 getStochasticGradient = boost::bind(&getGradientWrapper<Wave, Walker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
  functor2 getStochasticGradientMetric = boost::bind(&getGradientWrapper<Wave, Walker>::getMetric, &wrapper, _1, _2, _3, _4, _5, _6, _7, schd.deterministic);
  //functor0 getStochasticGradientVariance = boost::bind(&getGradientWrapper<Wave, Walker>::getVariance, &wrapper, _1, _2, _3, _4, _5, _6, _7, schd.deterministic);
  functor5 getStochasticGradientHessian = boost::bind(&getGradientWrapper<Wave, Walker>::getHessian, &wrapper, _1, _2, _3, _4, _5, _6, _7, schd.deterministic);
  functor6 getStochasticGradientHessianDirect = boost::bind(&getGradientWrapper<Wave, Walker>::getHessianDirect, &wrapper, _1, _2, _3, _4, _5, _6, schd.deterministic);

  CorrSampleWrapper<Wave, Walker> wrap(schd.CorrSampleFrac * schd.stochasticIter);
  CorrSampleFunctor runCorrelatedSampling = boost::bind(&CorrSampleWrapper<Wave, Walker>::run, &wrap, _1, _2);

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

  VectorXd vars; wave.getVariables(vars);
  getGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
  
  if (commrank == 0)
  {
    cout << "Number of Jastrow vars: " << wave.getNumJastrowVariables() << endl;
    cout << "Number of Reference vars: " << wave.getNumVariables() - wave.getNumJastrowVariables() << endl;
  }

  functor3 getStochasticGradientRealSpace = boost::bind(&getGradientWrapper<Wave, Walker>::getGradientRealSpace, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
  functor2 getStochasticGradientMetricRealSpace = boost::bind(&getGradientWrapper<Wave, Walker>::getMetricRealSpace, &wrapper, _1, _2, _3, _4, _5, _6, _7, schd.deterministic);
  functor5 getStochasticGradientHessianRealSpace = boost::bind(&getGradientWrapper<Wave, Walker>::getHessianRealSpace, &wrapper, _1, _2, _3, _4, _5, _6, _7, schd.deterministic);
  functor6 getStochasticGradientHessianDirectRealSpace = boost::bind(&getGradientWrapper<Wave, Walker>::getHessianDirectRealSpace, &wrapper, _1, _2, _3, _4, _5, _6, schd.deterministic);

  //CorrSampleWrapper<Wave, Walker> wrap(0.15 * schd.stochasticIter);
  CorrSampleWrapper<Wave, Walker> wrap(schd.CorrSampleFrac * schd.stochasticIter);
  CorrSampleFunctor runCorrelatedSamplingRealSpace = boost::bind(&CorrSampleWrapper<Wave, Walker>::runRealSpace, &wrap, _1, _2);

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
      if (schd.direct) {
        directLM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay, schd.sgdIter);
        optimizer.optimize(vars, getStochasticGradientHessianDirectRealSpace, runCorrelatedSamplingRealSpace, schd.restart);
      }
      else {  
        LM optimizer(schd.maxIter, schd.stepsizes, schd.hDiagShift, schd.sDiagShift, schd.decay);
        optimizer.optimize(vars, getStochasticGradientHessianRealSpace, runCorrelatedSamplingRealSpace, schd.restart);
      }
    }
  }
  if (schd.printVars && commrank==0) wave.printVariables();
}
