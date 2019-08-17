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

#include <Eigen/Dense>
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>

using namespace Eigen;


class Residuals;

void optimizeJastrowParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func,
    Residuals& residual);

void optimizeOrbitalParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func,
    Residuals& residual);


