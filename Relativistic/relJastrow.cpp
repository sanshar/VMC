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
#include "relJastrow.h"
#include "Correlator.h"
#include "relDeterminants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"

using namespace Eigen;

relJastrow::relJastrow () {    
  int norbs = relDeterminant::norbs;
  SpinCorrelator = MatrixXd::Constant(2*norbs, 2*norbs, 1.);

/*
  if (schd.optimizeCps)
    SpinCorrelator += 0.1 * MatrixXd::Random(2*norbs, 2*norbs);
*/
  bool readJastrow = false;
  char file[5000];
  sprintf(file, "Jastrow.txt");
  ifstream ofile(file);
  if (ofile)
    readJastrow = true;
  if (readJastrow) {
    for (int i = 0; i < SpinCorrelator.rows(); i++) {
      for (int j = 0; j < SpinCorrelator.rows(); j++){
        ofile >> SpinCorrelator(i, j);
      }
    }
  }
};


double relJastrow::Overlap(const relDeterminant &d) const
{
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  double ovlp = 1.0;
  for (int i=0; i<closed.size(); i++) {
    for (int j=0; j<=i; j++) {
      int I = max(closed[i], closed[j]), J = min(closed[i], closed[j]);

      ovlp *= SpinCorrelator(I, J);
    }
  }
  return ovlp;
}


double relJastrow::OverlapRatio (const relDeterminant &d1, const relDeterminant &d2) const {
  return Overlap(d1)/Overlap(d2);
}


double relJastrow::OverlapRatio(int i, int a, const relDeterminant &dcopy, const relDeterminant &d) const
{
  return OverlapRatio(dcopy, d);
}

double relJastrow::OverlapRatio(int i, int j, int a, int b, const relDeterminant &dcopy, const relDeterminant &d) const
{
  return OverlapRatio(dcopy, d);
}



void relJastrow::OverlapWithGradient(const relDeterminant& d, 
                              VectorXd& grad,
                              const double& ovlp) const {
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  if (schd.optimizeCps) {
    for (int i=0; i<closed.size(); i++) {
      for (int j=0; j<=i; j++) {
        int I = max(closed[i], closed[j]), J = min(closed[i], closed[j]);
        grad[I*(I+1)/2 + J] += ovlp/SpinCorrelator(I, J);
      }
    }
  }
}

long relJastrow::getNumVariables() const
{
  long spinOrbs = SpinCorrelator.rows();
  return spinOrbs*(spinOrbs+1)/2;
}


void relJastrow::getVariables(Eigen::VectorXd &v) const
{
  int numVars = 0;
  for (int i=0; i<SpinCorrelator.rows(); i++)
    for (int j=0; j<=i; j++) {
      v[numVars] = SpinCorrelator(i,j);
      numVars++;
    }
}

void relJastrow::updateVariables(const Eigen::VectorXd &v)
{
  int numVars = 0;
  for (int i=0; i<SpinCorrelator.rows(); i++)
    for (int j=0; j<=i; j++) {
      SpinCorrelator(i,j) = v[numVars];
      numVars++;
    }

}

void relJastrow::printVariables() const
{
  cout << "Jastrow"<< endl;
  //for (int i=0; i<SpinCorrelator.rows(); i++)
  //  for (int j=0; j<=i; j++)
  //    cout << SpinCorrelator(i,j);
  cout << SpinCorrelator << endl << endl;
}
