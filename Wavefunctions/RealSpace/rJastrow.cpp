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
  //int Q=2;
  int N = 7;
  vector<int> I(N), J(N), m(N), n(N), o(N), ss(N), fixed(N);

  _params.resize(N, 0); _gradHelper.resize(N, 0.);
  int Natom = schd.Ncharge.size();


  //EE 3 terms
  I[0] = -1; J[0] = -1; m[0] = 0; n[0] = 0; o[0] = 1; ss[0] = 1; fixed[0] = 1; _params[0] = 0.25;
  I[1] = -1; J[1] = -1; m[1] = 0; n[1] = 0; o[1] = 1; ss[1] = 0; fixed[1] = 1; _params[1] = 0.5;

  I[2] = -1; J[2] = -1; m[2] = 0; n[2] = 0; o[2] = 2; ss[2] = 1; fixed[2] = 0; _params[2] = 0.;
  I[3] = -1; J[3] = -1; m[3] = 0; n[3] = 0; o[3] = 2; ss[3] = 0; fixed[3] = 0; _params[3] = 0.;

  I[4] = -1; J[4] = -1; m[4] = 0; n[4] = 0; o[4] = 3; ss[4] = 1; fixed[4] = 0; _params[4] = 0.;
  I[5] = -1; J[5] = -1; m[5] = 0; n[5] = 0; o[5] = 3; ss[5] = 0; fixed[5] = 0; _params[5] = 0.;
  
  //NE 3 terms
  I[6] =  0; J[6] =  0; m[6] = 1; n[6] = 1; o[6] = 0; ss[6] = 2; fixed[6] = 0; _params[6] = 0.;
  //I[7] =  0; J[7] =  0; m[7] = 2; n[7] = 1; o[7] = 0; ss[7] = 0; fixed[7] = 0; _params[7] = 0.;
  //I[8] =  0; J[8] =  0; m[8] = 3; n[8] = 1; o[8] = 0; ss[8] = 0; fixed[8] = 0; _params[8] = 0.;


  _jastrow.I     = I;
  _jastrow.J     = J;
  _jastrow.m     = m;
  _jastrow.n     = n;
  _jastrow.o     = o;
  _jastrow.ss    = ss;
  _jastrow.fixed = fixed;
};


double rJastrow::exponential(const rDeterminant& d) {
  double exponent = 0.;
  _jastrow.exponential(d, &_params[0], &_gradHelper[0]);
  return exponent;
}

double rJastrow::exponentDiff(int i, Vector3d& coord, const rDeterminant& d) {
  return _jastrow.exponentDiff(i, coord, d, &_params[0], &_gradHelper[0]);
}

void rJastrow::UpdateGradientAndExponent(MatrixXd& Gradient,
                                         const MatrixXd& Rij,
                                         const MatrixXd& RiN,
                                         const rDeterminant& d,
                                         const Vector3d& oldCoord, int i) const {
  _jastrow.UpdateGradient(Gradient, d, oldCoord, i, &_params[0]
                            , const_cast<double *>(&_gradHelper[0]));
}

void rJastrow::UpdateLaplacian(VectorXd& laplacian,
                               const MatrixXd& Rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord, int i) const {
  _jastrow.UpdateLaplacian(laplacian, d, oldCoord, i, &_params[0]);
}

void rJastrow::InitGradient(MatrixXd& Gradient,
                            const MatrixXd& Rij,
                            const MatrixXd& RiN,
                            const rDeterminant& d) const {
  _jastrow.InitGradient(Gradient, d, &_params[0]);

}

void rJastrow::InitLaplacian(VectorXd& laplacian,
                             const MatrixXd& Rij,
                             const MatrixXd& RiN,
                             const rDeterminant& d) const {
  _jastrow.InitLaplacian(laplacian, d, &_params[0]);
}

long rJastrow::getNumVariables() const
{
  int numVars = _params.size();
  return numVars;
}


void rJastrow::getVariables(Eigen::VectorXd &v) const
{
  for (int t=0; t<_params.size(); t++)
    v[t] = _params[t];
}

void rJastrow::updateVariables(const Eigen::VectorXd &v)
{
  for (int t=0; t<_params.size(); t++) {
    _params[t] = v[t];
  }

}

void rJastrow::OverlapWithGradient(VectorXd& grad) const {
  int index = 0;
  _jastrow.OverlapWithGradient(grad, index, _gradHelper);
}


void rJastrow::printVariables() const
{
  //for (int t=0; t<params.size(); t++)
  //cout << params[t] ;
  cout << endl;
}


