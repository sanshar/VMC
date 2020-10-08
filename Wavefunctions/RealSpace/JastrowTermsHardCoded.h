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

#ifndef RJastrowTermsHardCoded_HEADER_H
#define RJastrowTermsHardCoded_HEADER_H


#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "rDeterminants.h"
#include "boost/serialization/export.hpp"

using namespace std;
using namespace Eigen;


bool electronsOfCorrectSpin(const int& i, const int& j, const int& ss);

void scaledRij(double& rij, double& rijbar,
               double& df, double& d2f);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowEEValue(int i, int j, int maxQ,
                      const vector<Vector3d>& r,
                      const VectorXd& params,
                      int startIndex,
                      int ss) ;

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowEEValueGrad(int i, int j, int maxQ,
                          const vector<Vector3d>& r,
                          Vector3d& grad,
                          const VectorXd& params,
                          int startIndex,
                          int ss);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values
void JastrowEEValues(int i, int j, int maxQ, const vector<Vector3d>& r, VectorXd& values, double factor, int startIndex, int ss);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
void JastrowEE(int i, int j, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex,
               int ss);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowENValue(int i, int maxQ,
                      const vector<Vector3d>& r,
                      const VectorXd& params,
                      int startIndex) ;

double JastrowENValueGrad(int i, int maxQ,
                          const vector<Vector3d>& r,
                          Vector3d& grad,
                          const VectorXd& params,
                          int startIndex);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values
void JastrowENValues(int i, int maxQ, const vector<Vector3d>& r, VectorXd& values, double factor, int startIndex);

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
void JastrowEN(int i, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex) ;

// (riN^m  jN^n + rjN^m riN^n) rij^o
// m + n + o <= maxQ
//returns an array of values, gx, gy, gz
double JastrowEENValue(int i, int j, int maxQ,
                       const vector<Vector3d>& r,
                       const VectorXd& params,
                       int startIndex,
                       int ss) ;

double JastrowEENValueGrad(int i, int j, int maxQ,
                         const vector<Vector3d>& r,
                         Vector3d& grad,
                         const VectorXd& params,
                         int startIndex,
                         int ss);

// (riN^m  jN^n + rjN^m riN^n) rij^o
// m + n + o <= maxQ
//returns an array of values
void JastrowEENValues(int i, int j, int maxQ, const vector<Vector3d>& r, VectorXd& values, double factor, int startIndex, int ss);

// (riN^m  jN^n + rjN^m riN^n) rij^o
// m + n + o <= maxQ
//returns an array of values, gx, gy, gz
void JastrowEEN(int i, int j, int maxQ,
                const vector<Vector3d>& r,
                VectorXd& values, MatrixXd& gx,
                MatrixXd& gy, MatrixXd& gz,
                MatrixXd& laplace, double factor,
                int startIndex,
                int ss);


//initializes N_I = \sum_i n_I (r_i) vector and n_I (r_i) matrix for four body jastrows
void JastrowEENNinit(const vector<Vector3d> &r, VectorXd &N, MatrixXd &n, std::array<MatrixXd, 3> &gradn, MatrixXd &lapn);

//updates N_I = \sum_i n_I (r_i) vector and n_I (r_i) matrix for four body jastrows, assumes r has been updated
void JastrowEENNupdate(int elec, const vector<Vector3d> &r, VectorXd &N, MatrixXd &n, std::array<MatrixXd, 3> &gradn, MatrixXd &lapn);

//returns vector of param values
void JastrowEENNValues(const VectorXd &N, const MatrixXd &n, VectorXd &ParamValues, int startIndex);

//returns gradient with respect to electron coordinates
void JastrowEENNgradient(int elec, const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, Vector3d &grad, const VectorXd &params, int startIndex);

//returns overlap ratio assuming elec is moved to coord
double JastrowEENNfactor(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, const VectorXd &params, int startIndex);

//returns overlap ratio and values assuming elec is moved to coord
double JastrowEENNfactorVector(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, VectorXd &ParamValues, int startIndex);

//returns overlap ratio and gradient with respect to electron coordinates assuming elec is moved to coord
double JastrowEENNfactorAndGradient(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, Vector3d &grad, const VectorXd &params, int startIndex);

//populates gx, gy, gz, values, and laplacian
void JastrowEENN(const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, const MatrixXd &lapn, VectorXd &ParamValues, std::array<MatrixXd, 3> &ParamGradient, MatrixXd &ParamLaplacian, int startIndex);

#endif
