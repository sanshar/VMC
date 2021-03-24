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

#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "igl/slice.h"
#include "igl/slice_into.h"

#include "Determinants.h"
#include "rDeterminants.h"
#include "rMultiSlater.h"
#include "global.h"
#include "input.h"


using namespace Eigen;

rMultiSlater::rMultiSlater() 
{
  initHforbs();
  initCiExpansion();
}

void rMultiSlater::initHforbs() 
{
  int norbs = Determinant::norbs;
  int size; //dimension of the mo coeff matrix
  if (schd.hf == "rhf" || schd.hf == "uhf") {
    size = norbs;
  }
  else if (schd.hf == "ghf") {
    //multi slater ghf not supported
  }

  HforbsA = MatrixXcd::Zero(size, size);
  HforbsB = MatrixXcd::Zero(size, size);
  readHF(HforbsA, HforbsB, schd.hf);

  //multi slater code only implemented for real CI wavefunctions
  /*
  if (schd.ifComplex && HforbsA.imag().isZero(0) == 0 && HforbsB.imag().isZero(0) == 0)
  {
    HforbsA.imag() = 0.01 * MatrixXd::Random(size, size);
    HforbsB.imag() = 0.01 * MatrixXd::Random(size, size);
  }
  else if (!schd.ifComplex)
  {
    HforbsA.imag() = MatrixXd::Zero(size, size);
    HforbsB.imag() = MatrixXd::Zero(size, size);
  }
  */
}

void rMultiSlater::initCiExpansion()
{
  string fname = "dets";
  readDeterminants(fname, ref, ciExcitations, ciParity, ciCoeffs);

  //open orbitals
  int norbs = Determinant::norbs;
  if (schd.nciAct > 0) { norbs = schd.nciAct; }
  for (int i = 0; i < norbs; i++)
  {
    if (std::find(ref[0].begin(), ref[0].end(), i) == ref[0].end()) { open[0].push_back(i); }
    if (std::find(ref[1].begin(), ref[1].end(), i) == ref[1].end()) { open[1].push_back(i); }
  }

  //number of configurations in expansion
  numDets = ciCoeffs.size();

  //internal indices of excitations
  for (int I = 0; I < getNumOfDets(); I++)
  {
    //alpha
    const Eigen::VectorXi &desA = ciExcitations[0][I][0];
    const Eigen::VectorXi &creA = ciExcitations[0][I][1];

    std::vector<int> rowA;
    for (int i = 0; i < desA.size(); i++) { rowA.push_back(std::find(ref[0].begin(), ref[0].end(), desA(i)) - ref[0].begin()); }
    std::vector<int> colA;
    for (int i = 0; i < creA.size(); i++) { colA.push_back(std::find(open[0].begin(), open[0].end(), creA(i)) - open[0].begin()); }

    Eigen::Map<VectorXi> rowVecA(rowA.data(), rowA.size());
    Eigen::Map<VectorXi> colVecA(colA.data(), colA.size());

    std::array<VectorXi, 2> indicesA;
    indicesA[0] = rowVecA;
    indicesA[1] = colVecA;

    ciIndices[0].push_back(indicesA);

    //beta
    const Eigen::VectorXi &desB = ciExcitations[1][I][0];
    const Eigen::VectorXi &creB = ciExcitations[1][I][1];

    std::vector<int> rowB;
    for (int i = 0; i < desB.size(); i++) { rowB.push_back(std::find(ref[1].begin(), ref[1].end(), desB(i)) - ref[1].begin()); }
    std::vector<int> colB;
    for (int i = 0; i < creB.size(); i++) { colB.push_back(std::find(open[1].begin(), open[1].end(), creB(i)) - open[1].begin()); }

    Eigen::Map<VectorXi> rowVecB(rowB.data(), rowB.size());
    Eigen::Map<VectorXi> colVecB(colB.data(), colB.size());

    std::array<VectorXi, 2> indicesB;
    indicesB[0] = rowVecB;
    indicesB[1] = colVecB;

    ciIndices[1].push_back(indicesB);
  }
}

/*
void rMultiSlater::getVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const
{
  int norbs = Determinant::norbs;
  for (size_t i = 0; i < numDets; i++) { v[i] = ciCoeffs[i]; }

  if (schd.hf == "rhf" || schd.hf == "uhf") {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        v[numDets + i * norbs + j] = HforbsA(i, j).real();
        //v[numDets + 4 * i * norbs + 2 * j + 1] = HforbsA(i, j).imag();
      }
    }
  }

  if (schd.hf == "uhf") {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        v[numDets + norbs * norbs + i * norbs + j] = HforbsB(i, j).real();
        //v[numDets + 2 * norbs * norbs + 4 * i * norbs + 2 * j + 1] = HforbsB(i, j).imag();
      }
    }
  }

}

void rMultiSlater::updateVariables(Eigen::VectorBlock<Eigen::VectorXd> &v)
{
  int norbs = Determinant::norbs;
  for (size_t i = 0; i < numDets; i++) { ciCoeffs[i] = v[i]; }

  if (schd.hf == "rhf" || schd.hf == "uhf") {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        HforbsA(i, j) = std::complex<double>(v[numDets + i * norbs + j], 0.0);
        //HforbsA(i, j) = std::complex<double>(v[numDets + 4 * i * norbs + 2 * j], v[numDets + 4 * i * norbs + 2 * j + 1]);
        if (schd.hf != "uhf") { HforbsB(i, j) = HforbsA(i, j); }
      }
    }
  }

  if (schd.hf == "uhf") {
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        HforbsB(i, j) = std::complex<double>(v[numDets + norbs * norbs + i * norbs + j], 0.0);
        //HforbsB(i, j) = std::complex<double>(v[numDets + 2 * norbs * norbs + 4 * i * norbs + 2 * j], v[numDets + 2 * norbs * norbs + 4 * i * norbs + 2 * j + 1]);
      }
    }
  }

}
*/

void rMultiSlater::printVariables() const
{
  cout << "\nciCoeffs\n";
  for (size_t i = 0; i < numDets; i++) { cout << ciCoeffs[i] << endl; }

  cout << "\nHforbsA\n";
  for (int i = 0; i < HforbsA.rows(); i++) {
    for (int j = 0; j < HforbsA.rows(); j++) cout << "  " << HforbsA(i, j);
    cout << endl;
  }

  /*
  if (schd.hf == "uhf") {
    cout << "\nHforbsB\n";
    for (int i = 0; i < HforbsB.rows(); i++) {
      for (int j = 0; j < HforbsB.rows(); j++) cout << "  " << HforbsB(i, j);
      cout << endl;
    }
  }
  */
  cout << endl;
}
