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
    if (o == 1 && schd.enforceEECusp) continue;
    EENterms++;
  }
  EENoppositeSpinIndex = EENsameSpinIndex + schd.uniqueAtoms.size()*EENterms;

   //4body Jastrows
   int norbs = schd.basis->getNorbs();
   int numSorb = 0;
   int numPorb = 0;
   for (int i = 0; i < schd.NSbasis.size(); i++) { numSorb += schd.NSbasis[i].size(); }
   for (int i = 0; i < schd.NPbasis.size(); i++) { numPorb += schd.NPbasis[i].size(); }
   EENNlinearIndex = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;
   //EENNlinearIndex = EENsameSpinIndex;
   if (schd.fourBodyJastrowBasis == NC || schd.fourBodyJastrowBasis == sNC) {
     EENNIndex = EENNlinearIndex + 2 * schd.Ncharge.size();
   }
   else if (schd.fourBodyJastrowBasis == SG) {
     EENNIndex = EENNlinearIndex + 2 * schd.Ncharge.size();
   }
   else if (schd.fourBodyJastrowBasis == AB2) {
     EENNIndex = EENNlinearIndex + 2 * norbs;
   }
   else if (schd.fourBodyJastrowBasis == sAB2) {
     EENNIndex = EENNlinearIndex + 2 * numSorb;
   }
   else if (schd.fourBodyJastrowBasis == spAB2) {
     EENNIndex = EENNlinearIndex + 2 * (numSorb + numPorb);
   }
   else if (schd.fourBodyJastrowBasis == asAB2) {
     EENNIndex = EENNlinearIndex + 2 * schd.asAO.size();
   }
   else if (schd.fourBodyJastrowBasis == SS) {
     EENNIndex = EENNlinearIndex + 2 * norbs;
   }
   else if (schd.fourBodyJastrowBasis == G) {
     EENNIndex = EENNlinearIndex + 2 * schd.gridGaussians.size();
   }
   
   int numParams = EENoppositeSpinIndex + EENoppositeSpinIndex - EENsameSpinIndex;

   if (schd.fourBodyJastrow) {
     if (schd.fourBodyJastrowBasis == NC || schd.fourBodyJastrowBasis == sNC) {
       numParams = EENNIndex + (2 * schd.Ncharge.size()) * (2 * schd.Ncharge.size() + 1) / 2;
     }
     else if (schd.fourBodyJastrowBasis == SG) {
       numParams = EENNIndex + (2 * schd.Ncharge.size()) * (2 * schd.Ncharge.size() + 1) / 2;
     }
     else if(schd.fourBodyJastrowBasis == AB2) {
       numParams = EENNIndex + (2 * norbs) * (2 * norbs + 1) / 2;
     }
     else if(schd.fourBodyJastrowBasis == sAB2) {
       numParams = EENNIndex + (2 * numSorb) * (2 * numSorb + 1) / 2;
     }
     else if (schd.fourBodyJastrowBasis == spAB2) {
       numParams = EENNIndex + (2 * (numSorb + numPorb)) * (2 * (numSorb + numPorb) + 1) / 2;
     }
     else if (schd.fourBodyJastrowBasis == asAB2) {
       numParams = EENNIndex + (2 * schd.asAO.size()) * (2 * schd.asAO.size() + 1) / 2;
     }
     else if(schd.fourBodyJastrowBasis == SS) {
       numParams = EENNIndex + (2 * norbs) * (2 * norbs + 1) / 2;
     }
     else if (schd.fourBodyJastrowBasis == G) {
       numParams = EENNIndex + (2 * schd.gridGaussians.size()) * (2 * schd.gridGaussians.size() + 1) / 2;
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

  //_params.resize(numParams, 1.e-4);
  _params.resize(numParams, 0.0);

  _params[EEsameSpinIndex] = 0.25;
  _params[EEoppositeSpinIndex] = 0.5;
  if (schd.noENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { _params[ENIndex + I * Qmax] = 0.0; }
  if (schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { _params[ENIndex + I * Qmax] = - schd.uniqueAtoms[I]; }

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

  //ensure these values stay constant
  _params[EEsameSpinIndex] = 0.25;
  _params[EEoppositeSpinIndex] = 0.5;
  if (schd.noENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { _params[ENIndex + I * Qmax] = 0.0; }
  if (schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { _params[ENIndex + I * Qmax] = - schd.uniqueAtoms[I]; }
}


void rJastrow::printVariables() const
{
  int EENterms = 0;
  for (int m = 1; m <= QmaxEEN; m++) 
  for (int n = 0; n <= m; n++)
  for (int o = 0; o <= (QmaxEEN-m-n); o++) {
    if (n == 0 && o == 0) continue; //EN term
    if (o == 1 && schd.enforceEECusp) continue;
    EENterms++;
  }

  cout << endl;
  cout << "__rJastrow parameters__" << endl << endl;

  cout << "EEsameSpin" << endl;
  for (int t=EEsameSpinIndex; t<EEoppositeSpinIndex; t++)
    cout << _params[t] << " ";
  cout << endl << endl;

  cout << "EEoppositeSpin" << endl;
  for (int t=EEoppositeSpinIndex; t<ENIndex; t++)
    cout << _params[t] << " ";
  cout << endl << endl;

  cout << "EN" << endl;
  for (int I = 0; I < schd.uniqueAtoms.size(); I++) {
    cout << schd.uniqueAtoms[I] << " | ";
    for (int i = 0; i < Qmax; i++) {
      cout << _params[ENIndex + Qmax * I + i] << " ";
    }
    cout << endl;
  }
  cout << endl;

  cout << "EENsameSpin" << endl;
  for (int I = 0; I < schd.uniqueAtoms.size(); I++) {
    cout << schd.uniqueAtoms[I] << " | ";
    for (int i = 0; i < EENterms; i++) {
      cout << _params[EENsameSpinIndex + EENterms * I + i] << " ";
    }
    cout << endl;
  }
  cout << endl;

  cout << "EENoppositeSpin" << endl;
  for (int I = 0; I < schd.uniqueAtoms.size(); I++) {
    cout << schd.uniqueAtoms[I] << " | ";
    for (int i = 0; i < EENterms; i++) {
      cout << _params[EENoppositeSpinIndex + EENterms * I + i] << " ";
    }
    cout << endl;
  }
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
        int max = std::max(i, j);
        int min = std::min(i, j);
        cout << _params[EENNlinearIndex + size + max * (max + 1) / 2 + min] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  cout << "rJastrow.txt" << endl;
  for (int t=0; t<_params.size(); t++)
    cout << _params[t]<<"  " ;
  cout << endl;
}


