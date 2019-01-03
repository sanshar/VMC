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

/*
rJastrow::rJastrow () {
  //int Q=2;
  int N = 3;
  vector<int> I(N), J(N), m(N), n(N), o(N), ss(N), fixed(N);

  _params.resize(N, 0); _gradHelper.resize(N, 0.);
  int Natom = schd.Ncharge.size();


  //EE 3 terms
  I[0] = 0; J[0] = 0; m[0] = 0; n[0] = 0; o[0] = 1; ss[0] = 1; fixed[0] = 1; _params[0] = 0.25;
  I[1] = 0; J[1] = 0; m[1] = 0; n[1] = 0; o[1] = 1; ss[1] = 0; fixed[1] = 1; _params[1] = 0.5;

  I[2] = 0; J[2] = 0; m[2] = 1; n[2] = 0; o[2] = 0; ss[2] = 2; fixed[2] = 0; _params[2] = 0.01;
  //I[3] = -1; J[3] = -1; m[3] = 0; n[3] = 0; o[3] = 2; ss[3] = 0; fixed[3] = 0; _params[3] = 0.01;

  //I[4] = -1; J[4] = -1; m[4] = 0; n[4] = 0; o[4] = 3; ss[4] = 1; fixed[4] = 0; _params[4] = 0.01;
  //I[5] = -1; J[5] = -1; m[5] = 0; n[5] = 0; o[5] = 3; ss[5] = 0; fixed[5] = 0; _params[5] = 0.01;
  
  //NE 3 terms
  //I[6] =  0; J[6] =  0; m[6] = 1; n[6] = 1; o[6] = 0; ss[6] = 2; fixed[6] = 0; _params[6] = 0.01;
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
*/


rJastrow::rJastrow () {
  int Qmax=5;

  vector<int> I, J, m, n, o, ss, fixed;
  int Natom = schd.Ncharge.size();

  for (int Q=1; Q<=Qmax; Q++) {
    
    for (int mi=0; mi<=Q; mi++)
    for (int ni=0; ni<=Q; ni++)
    for (int oi=0; oi<=Q; oi++) {

      if (mi+ni+oi != Q || mi < ni) continue;
      
      for (int Ii = 0; Ii < Natom; Ii++)
      for (int Ji = 0; Ji < Natom; Ji++) {
        I.push_back(Ii);
        J.push_back(Ji);
        m.push_back(mi);
        n.push_back(ni);
        o.push_back(oi);

        //special case of R12 terms with fixed amplitudes
        if (mi == 0 && ni == 0 && oi == 1) {
          fixed.push_back(1);
          ss.push_back(0);
          
          I.push_back(Ii);
          J.push_back(Ji);
          m.push_back(mi);
          n.push_back(ni);
          o.push_back(oi);
          fixed.push_back(1);
          ss.push_back(1);
        }
        else if (oi != 0 || (mi != 0 && ni != 0)) { //two electron term
          fixed.push_back(0);
          ss.push_back(0);
          
          I.push_back(Ii);
          J.push_back(Ji);
          m.push_back(mi);
          n.push_back(ni);
          o.push_back(oi);
          fixed.push_back(0);
          ss.push_back(1);
        }
        else {
          ss.push_back(2);
          fixed.push_back(0);
        }
      }

    }

  }

  int N = I.size();
  if (commrank == 0) cout << "Num Jastrow terms "<<N<<endl;
  _params.resize(N, 0); _gradHelper.resize(N, 0.);
  for (int i=0; i< I.size(); i++) {
    if (m[i] == 0 && n[i] == 0 && o[i] == 1 && ss[i] == 0)
      _params[i] = 0.5;
    else if (m[i] == 0 && n[i] == 0 && o[i] == 1 && ss[i] == 1)
      _params[i] = 0.25;
  }



  _jastrow.I     = I;
  _jastrow.J     = J;
  _jastrow.m     = m;
  _jastrow.n     = n;
  _jastrow.o     = o;
  _jastrow.ss    = ss;
  _jastrow.fixed = fixed;
};


double rJastrow::exponentialInitLaplaceGrad(const rDeterminant& d,
                                            MatrixXd& Gradient,
                                            VectorXd& laplacian) {
  double exponent = 0.;
  _jastrow.exponentialInitLaplaceGrad(d, Gradient, laplacian,
                                      &_params[0], &_gradHelper[0]);
  return exponent;
}

double rJastrow::exponentDiff(int i, Vector3d& coord, const rDeterminant& d) {
  return _jastrow.exponentDiff(i, coord, d, &_params[0], &_gradHelper[0]);
}

void rJastrow::UpdateLaplaceGrad(MatrixXd& Gradient,
                                 VectorXd& laplacian,
                                 const MatrixXd& Rij,
                                 const MatrixXd& RiN,
                                 const rDeterminant& d,
                                 const Vector3d& oldCoord, int i) const {

  _jastrow.UpdateLaplaceGrad(Gradient, laplacian, d, oldCoord, i, &_params[0]
                            , const_cast<double *>(&_gradHelper[0]));
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


