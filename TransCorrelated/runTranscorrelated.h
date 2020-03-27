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
#include "TransevaluateE.h"


template<typename Wave>
void runTranscorrelated(Wave& wave, bool isGutzwiller = false) {

  if (schd.restart || schd.fullRestart)
    wave.readWave();
  
  if (commrank == 0)
  {
    cout << "Number of Jastrow vars: " << wave.getCorr().getNumVariables() << endl;
    cout << "Number of Reference vars: " << wave.getNumVariables() - wave.getCorr().getNumVariables() << endl;
  }
  
  getTranscorrelationWrapper<Wave> wrapper(wave);

  if (schd.wavefunctionType == "NOCI")
    wrapper.optimizeWavefunctionNOCI(schd.nNociSlater);
  else if (isGutzwiller)
    wrapper.optimizeWavefunctionGJ();
  else
    wrapper.optimizeWavefunction();
  
  if (schd.printVars && commrank==0) wave.printVariables();
  
}
