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
using functorPen = boost::function<double (VectorXd&, VectorXd&, std::complex<double>&, double&, double&, double&)>;
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
  }
  // energy only, no optimization
  if (schd.onlyEne == true && schd.excitedState==0){
    if (commrank == 0) cout << "Relativistic energy calculation" << endl;
    relGetEnergyWrapper<Wave, Walker> wrapper(wave, walk, schd.deterministic);
    functor2r relGetEne = boost::bind(&relGetEnergyWrapper<Wave, Walker>::getEnergy, &wrapper, _1, _2, _3, _4);
    eneOnly eo(schd.deterministic);
    eo.relGetEnergy(vars, relGetEne);
  }
  // regular VMC calculation
  else if (schd.onlyEne == false && schd.excitedState==0 && (schd.method == amsgrad || schd.method == amsgrad_sgd)) {
    if (commrank == 0) cout << "Relativistic vmc calculation" << endl;
    relGetGradientWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc);
    functor1rc relGetGradient = boost::bind(&relGetGradientWrapper<Wave, Walker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
    if (schd.stepsizes.empty()) {
      relAMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      optimizer.optimize(vars, relGetGradient, schd.restart);
    }
    else {
      cout << "Not yet implemented for rel" << endl;
    }
  }
  // Determininstic excitedState calculation
  else if (schd.excitedState>0) {
    if (commrank == 0) cout << "Relativistic excited states vmc calculation" << endl;
    std::vector<relCorrelatedWavefunction<relJastrow, relSlater>> ls;
    for (int i=0; i<schd.excitedState; i++) {
      relCorrelatedWavefunction<relJastrow, relSlater> state;
      state.readWave(schd.excitedState-i-1);
      ls.push_back(state);
    }
    relGetGradientPenaltyWrapper<Wave, Walker> wrapper(wave, walk, schd.stochasticIter, schd.ctmc, ls);
    functor1rc relGetGradientPenalty = boost::bind(&relGetGradientPenaltyWrapper<Wave, Walker>::getGradientPenalty, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
    if (schd.stepsizes.empty()) {
      relAMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
      optimizer.optimize(vars, relGetGradientPenalty, schd.restart);
    }
  }
  else {
    cout << "No valid option selected for relativistic calculation" << endl;
  }
  
  if (schd.printVars && commrank==0) wave.printVariables();
  
}
