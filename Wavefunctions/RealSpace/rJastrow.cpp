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
  Terms.push_back(boost::shared_ptr<GeneralTerm>(new EEJastrow));
  Terms.push_back(boost::shared_ptr<GeneralTerm>(new ENJastrow));
};


long rJastrow::getNumVariables() const
{
  int numVars = 0;
  for (int t=0; t<Terms.size(); t++)
    numVars += Terms[t]->getNumVariables();
  return numVars;
}


void rJastrow::getVariables(Eigen::VectorXd &v) const
{
  int numVars = 0;
  for (int t=0; t<Terms.size(); t++)
    Terms[t]->getVariables(v, numVars);
}

void rJastrow::updateVariables(const Eigen::VectorXd &v)
{
  int numVars = 0;
  for (int t=0; t<Terms.size(); t++)
    Terms[t]->updateVariables(v, numVars);
}

void rJastrow::printVariables() const
{
  for (int t=0; t<Terms.size(); t++)
    Terms[t]->printVariables();
}


