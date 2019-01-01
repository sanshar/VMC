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

using namespace Eigen;
using namespace std;

rJastrow::rJastrow () {
  //the first EN term is not present
  //the first EE term is fixed
  ENorder = 1, EEorder = 5, EENorder = 0;
  int nNucleus = schd.Ncharge.size();
  
  _enparams.resize(nNucleus*(ENorder-1),0.);
  _envalues.resize(nNucleus*(ENorder-1),0.);

  _eeparams.resize(EEorder-1,0.);
  _eevalues.resize(EEorder-1,0.);
};


double rJastrow::exponential(MatrixXd& Rij, MatrixXd& RiN) {
  double exponent = 0.;
  exponent += _eejastrow.exponential(Rij, RiN, EEorder, &_eeparams[0], &_eevalues[0]); 
  exponent += _enjastrow.exponential(Rij, RiN, ENorder, &_enparams[0], &_envalues[0]);
  return exponent;
}

double rJastrow::exponentDiff(int i, Vector3d& coord, const rDeterminant& d) {
  double diff = 0.;
  diff += _eejastrow.exponentDiff(i, coord, d, EEorder, &_eeparams[0], &_eevalues[0]);
  diff += _enjastrow.exponentDiff(i, coord, d, ENorder, &_enparams[0], &_envalues[0]);
  return diff;  
}

void rJastrow::UpdateGradientAndExponent(MatrixXd& Gradient,
                                         const MatrixXd& Rij,
                                         const MatrixXd& RiN,
                                         const rDeterminant& d,
                                         const Vector3d& oldCoord, int i) const {
  _eejastrow.UpdateGradient(Gradient, Rij, RiN, d, oldCoord, i, EEorder, &_eeparams[0]
                            , const_cast<double *>(&_eevalues[0]));
  _enjastrow.UpdateGradient(Gradient, Rij, RiN, d, oldCoord, i, ENorder, &_enparams[0]
                            , const_cast<double *>(&_envalues[0]));
}

void rJastrow::UpdateLaplacian(VectorXd& laplacian,
                               const MatrixXd& Rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord, int i) const {
  _eejastrow.UpdateLaplacian(laplacian, Rij, RiN, d, oldCoord, i, EEorder, &_eeparams[0]);
  _enjastrow.UpdateLaplacian(laplacian, Rij, RiN, d, oldCoord, i, ENorder, &_enparams[0]);
}

void rJastrow::InitGradient(MatrixXd& Gradient,
                            const MatrixXd& Rij,
                            const MatrixXd& RiN,
                            const rDeterminant& d) const {
  _eejastrow.InitGradient(Gradient, Rij, RiN, d, EEorder, &_eeparams[0]);
  _enjastrow.InitGradient(Gradient, Rij, RiN, d, ENorder, &_enparams[0]);
}

void rJastrow::InitLaplacian(VectorXd& laplacian,
                             const MatrixXd& Rij,
                             const MatrixXd& RiN,
                             const rDeterminant& d) const {
  _eejastrow.InitLaplacian(laplacian, Rij, RiN, d, EEorder, &_eeparams[0]);
  _enjastrow.InitLaplacian(laplacian, Rij, RiN, d, ENorder, &_enparams[0]);
}

long rJastrow::getNumVariables() const
{
  int numVars = schd.Ncharge.size()*(ENorder-1) + (EEorder-1) ;
  return numVars;
}


void rJastrow::getVariables(Eigen::VectorXd &v) const
{
  for (int t=0; t<_eeparams.size(); t++)
    v[t] = _eeparams[t];

  for (int t=0; t<_enparams.size(); t++)
    v[t+EEorder-1] = _enparams[t];
}

void rJastrow::updateVariables(const Eigen::VectorXd &v)
{
  for (int t=0; t<_eeparams.size(); t++) {
    _eeparams[t] = v[t];
  }

  for (int t=0; t<_enparams.size(); t++)
    _enparams[t] = v[t+EEorder-1];

}

void rJastrow::OverlapWithGradient(VectorXd& grad) const {
  int index = 0;
  _eejastrow.OverlapWithGradient(grad, index, 2, EEorder, &_eevalues[0]);
  _enjastrow.OverlapWithGradient(grad, index, 2, ENorder, &_envalues[0]);
}


void rJastrow::printVariables() const
{
  //for (int t=0; t<params.size(); t++)
  //cout << params[t] ;
  cout << endl;
}


