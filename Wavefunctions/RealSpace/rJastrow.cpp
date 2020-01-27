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
#include "rJastrow.h"
#include "rDeterminants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"
#include <vector>
#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace std;


rJastrow::rJastrow () {
  Qmax = schd.Qmax;
  QmaxEEN = schd.QmaxEEN; //EEN jastrow is expensive -> use fewer powers

  EEsameSpinIndex = 0;
  EEoppositeSpinIndex = Qmax;
  ENIndex = 2*Qmax;
  EENsameSpinIndex = 2*Qmax + schd.Ncharge.size()*Qmax;
  
  //EENoppositeSpinIndex = EENsameSpinIndex;
  int EENterms = 0;
  for (int m = 1; m <= QmaxEEN; m++) 
  for (int n = 0; n <= m   ; n++) 
  for (int o = 0; o <= (QmaxEEN-m-n); o++) {
    if (n == 0 && o == 0) continue; //EN term
    EENterms++;
  }
  EENoppositeSpinIndex = EENsameSpinIndex + schd.Ncharge.size()*EENterms;

   //4body Jastrows
   EENNlinearIndex = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;
   EENNIndex = EENNlinearIndex + schd.Ncharge.size();
   
   int numParams = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;
   if (schd.fourBodyJastrow) {
       numParams = EENNIndex + schd.Ncharge.size() * (schd.Ncharge.size() + 1) / 2;
   }
   
   /*
   if (commrank == 0) {
   cout << "Num: " << numParams << endl;
   cout << "EEsameSpinIndex: " << EEsameSpinIndex << endl;
   cout << "EEoppositeSpinIndex: " << EEoppositeSpinIndex << endl;
   cout << "ENIndex: " << ENIndex << endl;
   cout << "EENsameSpinIndex: " << EENsameSpinIndex << endl;
   cout << "EENoppositeSpinIndex: " << EENoppositeSpinIndex << endl;
   cout << "EENNlinearIndex: " << EENNlinearIndex << endl;
   cout << "EENNIndex: " << EENNIndex << endl;
   }
   */ 

  //_params.resize(EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex, 1.e-4);
  _params.resize(numParams, 1.e-4);
  _params[EEsameSpinIndex] = 0.25;
  _params[EEoppositeSpinIndex] = 0.5;
  if (schd.optimizeCps == false) { _params.assign(_params.size(), 0.0); }
  //if (commrank == 0) cout << "Num Jastrow terms "<<_params.size()<<endl;
  //if rJastrow.txt file exists
  ifstream ifile("rJastrow.txt");
  if (ifile) {
      for (int i = 0; i < _params.size(); i++) { ifile >> _params[i]; }
  }
};


long rJastrow::getNumVariables() const
{
  if (!schd.optimizeCps) return 0;
  int numVars = _params.size();
  return numVars;
}


void rJastrow::getVariables(Eigen::VectorXd &v) const
{
  if (!schd.optimizeCps) return;
  for (int t=0; t<_params.size(); t++)
    v[t] = _params[t];
}

void rJastrow::updateVariables(const Eigen::VectorXd &v)
{
  if (!schd.optimizeCps) return;
  for (int t=0; t<_params.size(); t++) {
    _params[t] = v[t];
  }

}


void rJastrow::printVariables() const
{
  for (int t=0; t<_params.size(); t++)
    cout << _params[t]<<"  " ;
  cout << endl;
}


