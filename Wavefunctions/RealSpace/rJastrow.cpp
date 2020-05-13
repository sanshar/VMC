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
  EENsameSpinIndex = 2*Qmax + schd.uniqueAtoms.size()*Qmax;
  
  //EENoppositeSpinIndex = EENsameSpinIndex;
  int EENterms = 0;
  for (int m = 1; m <= QmaxEEN; m++) 
  for (int n = 0; n <= m   ; n++) 
  for (int o = 0; o <= (QmaxEEN-m-n); o++) {
    if (n == 0 && o == 0) continue; //EN term
    EENterms++;
  }
  EENoppositeSpinIndex = EENsameSpinIndex + schd.uniqueAtoms.size()*EENterms;

   //4body Jastrows
   int norbs = schd.basis->getNorbs();
   int numSorb = 0;
   for (int i = 0; i < schd.NSbasis.size(); i++) { numSorb += schd.NSbasis[i].size(); }
   EENNlinearIndex = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;
   //EENNlinearIndex = EENsameSpinIndex;
   if (schd.fourBodyJastrowBasis == NC || schd.fourBodyJastrowBasis == sNC) {
     EENNIndex = EENNlinearIndex + 2 * schd.Ncharge.size();
   }
   else if (schd.fourBodyJastrowBasis == AB) {
     EENNIndex = EENNlinearIndex + 2 * norbs;
   }
   else if (schd.fourBodyJastrowBasis == sAB) {
     EENNIndex = EENNlinearIndex + 2 * numSorb;
   }
   else if (schd.fourBodyJastrowBasis == SS) {
     EENNIndex = EENNlinearIndex + 2 * norbs;
   }
   
   int numParams = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;

   /*
   if (schd.fourBodyJastrow) {
     if (schd.fourBodyJastrowBasis == NC)
       numParams = EENNIndex + schd.Ncharge.size() * (schd.Ncharge.size() + 1) / 2;
     else if(schd.fourBodyJastrowBasis == AB)
       numParams = EENNIndex + norbs * (norbs + 1) / 2;
   }
   */
   if (schd.fourBodyJastrow) {
     if (schd.fourBodyJastrowBasis == NC || schd.fourBodyJastrowBasis == sNC) {
       numParams = EENNIndex + 4 * schd.Ncharge.size() * schd.Ncharge.size();
     }
     else if(schd.fourBodyJastrowBasis == AB) {
       numParams = EENNIndex + 4 * norbs * norbs;
     }
     else if(schd.fourBodyJastrowBasis == sAB) {
       numParams = EENNIndex + 4 * numSorb * numSorb;
     }
     else if(schd.fourBodyJastrowBasis == SS) {
       numParams = EENNIndex + 4 * norbs * norbs;
     }
   }
   
   /*
   if (commrank == 0) {
   cout << schd.fourBodyJastrowBasis << endl;
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
  //_params.resize(numParams, 0.0);
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
  cout << endl;
  cout << "__rJastrow parameters__" << endl;

  cout << "EEsamespin" << endl;
  for (int t=EEsameSpinIndex; t<EEoppositeSpinIndex; t++)
    cout << _params[t] << " ";
  cout << endl;

  cout << "EEoppositespin" << endl;
  for (int t=EEoppositeSpinIndex; t<ENIndex; t++)
    cout << _params[t] << " ";
  cout << endl;

  cout << "EN" << endl;
  int atm = 0;
  for (int t=ENIndex; t<EENsameSpinIndex; t++) {
    if (t % Qmax == 0) {
      cout << schd.uniqueAtoms[atm] << " | ";
      atm++;
    }
    cout << _params[t] << " ";
    if ((t + 1) % Qmax == 0) cout << endl;
  }

  cout << "EENsamespin" << endl;
  for (int t=EENsameSpinIndex; t<EENoppositeSpinIndex; t++)
    cout << _params[t] << " ";
  cout << endl;

  cout << "EENoppositespin" << endl;
  for (int t=EENoppositeSpinIndex; t<EENNlinearIndex; t++)
    cout << _params[t] << " ";
  cout << endl;

  if (schd.fourBodyJastrow == true) {
    int size = EENNIndex - EENNlinearIndex;

    cout << "EENNlinear" << endl;
    for (int i = 0; i < size; i++) {
        cout << _params[i + EENNlinearIndex] << " ";
    }
    cout << endl;

    cout << "EENNquadratic" << endl;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cout << _params[EENNlinearIndex + size + i * size + j] << " ";
      }
      cout << endl;
    }
  }

  cout << "rJastrow.txt" << endl;
  for (int t=0; t<_params.size(); t++)
    cout << _params[t]<<"  " ;
  cout << endl;
}


