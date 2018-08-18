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
#ifndef CIWavefunction_HEADER_H
#define CIWavefunction_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <array>

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class CPSSlaterWalker;

class Operator {
private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & cre
       & des
       & n;
  }

  public:

  std::array<short, 4> cre;
  std::array<short, 4> des;
  int n;

  Operator() : cre ({0,0,0,0}), des ({0,0,0,0}) { 
    n = 0;
  }

  //a1^\dag i1
  Operator(short a1, short i1) : cre ({a1}), des ({i1}) {
    n = 1;
  }

  //a2^\dag i2 a1^\dag i1
  Operator(short a1, short a2, short i1, short i2) :cre ({a2, a1}), des ({i2, i1}) {
    n = 2;
  }

};

/**
* This is the wavefunction, that extends a given wavefunction by doing
* a CI expansion on it
* |Psi> = \sum_i d_i O_i ||phi>
* where d_i are the coefficients and O_i are the list of operators
*/
template <typename Wfn, typename Walker>
class CIWavefunction
{
private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar  & wave
        & oplist
        & ciCoeffs;
  }

public:
  Wfn wave;
  std::vector<Operator> oplist;
  std::vector<double> ciCoeffs;

  CIWavefunction() {
    wave.read();
    oplist.resize(1);
    ciCoeffs.resize(1, 0.0);
  }

  void initWalker(Walker& walk) {
    wave.initWalker(walk);
  }  
  void initWalker(Walker& walk, Determinant& d) {
    wave.initWalker(walk, d);
  }  

  CIWavefunction(Wfn &w1, std::vector<Operator> &pop) : wave(w1), oplist(pop)
  {
    ciCoeffs.resize(oplist.size(), 0.0);
    ciCoeffs[0] = 1.0;
  };

  void getVariables(VectorXd& vars) {
    for (int i=0; i<ciCoeffs.size(); i++)
      vars[i] = ciCoeffs[i];
  }

  void updateVariables(VectorXd& vars) {
    for (int i=0; i<vars.rows(); i++)
      ciCoeffs[i] = vars[i];
  }

  long getNumVariables() {
    return ciCoeffs.size();
  }

  double Overlap(Walker& walk) {
    double totalovlp = 0.0;
    double ovlp0 = wave.Overlap(walk);

    for (int i = 0; i < oplist.size(); i++)
    {
      Determinant dcopy = walk.getDet();

      bool valid = true;
      for (int j = 0; j < oplist[i].n; j++)
      {
        if (dcopy.getocc(oplist[i].cre[j]) == true &&
            dcopy.getocc(oplist[i].des[j] == false))
        {
          dcopy.setocc(oplist[i].cre[j], false);
          dcopy.setocc(oplist[i].des[j], true);
        }
        else
        {
          valid = false;
          break;
        }
      }

      if (valid) {
        double ovlpdetcopy;
        int excitationDistance = dcopy.ExcitationDistance(walk.getDet());

        if (excitationDistance == 0) {
          ovlpdetcopy = 1.0;
        }
        //if there is a repeat of indices then sometimes a double
        //excitation can only result in excitaiton distance of 1
        if (excitationDistance == 1) {
          int c1 = -1, d1 =-1;
          if (dcopy.getocc(oplist[i].cre[0]) == false)
            c1 = oplist[i].cre[0];
          else
            c1 = oplist[i].cre[1];

          if (dcopy.getocc(oplist[i].des[0]) == true)
            d1 = oplist[i].des[0];
          else
            d1 = oplist[i].des[1];

          double ovlpdetcopy = wave.getJastrowFactor(c1 / 2, d1 / 2, dcopy, walk.d);
          if (c1%2 == 0)
            ovlpdetcopy *= walk.getDetFactorA(c1 / 2, d1 / 2, wave, false);
        }
        else if (excitationDistance == 2) {
          int I = oplist[i].cre[0], J = oplist[i].cre[1], A = oplist[i].des[0], B = oplist[i].des[1]; 
          double JastrowFactor = wave.getJastrowFactor(I / 2, J / 2, A / 2, B / 2, dcopy, walk.d);

          bool doparity = false;
          if (I % 2 == J % 2 && I % 2 == 0)
            ovlpdetcopy = walk.getDetFactorA(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else if (I % 2 == J % 2 && I % 2 == 1)
            ovlpdetcopy = walk.getDetFactorB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else if (I % 2 != J % 2 && I % 2 == 0)
            ovlpdetcopy = walk.getDetFactorAB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else
            ovlpdetcopy = walk.getDetFactorAB(J / 2, I / 2, B / 2, A / 2, *this, doparity) * JastrowFactor;
        }
        else {
          cout <<"higher than triple excitation not yet supported."<<endl;
          exit(0);
        }
        totalovlp += ciCoeffs[i] * ovlpdetcopy * ovlp0;
      }
    }
    return totalovlp;
  }

  double OverlapWithGradient(Walker &walk,
                           double &factor,
                           Eigen::VectorXd &grad)
  {
    VectorXd gradcopy = grad;
    gradcopy.setZero();

    double ovlp0 = wave.Overlap(walk);

    for (int i = 0; i < oplist.size(); i++)
    {
      Determinant dcopy = walk.getDet();

      bool valid = true;
      for (int j = 0; j < oplist[i].n; j++)
      {
        if (dcopy.getocc(oplist[i].cre[j]) == true &&
            dcopy.getocc(oplist[i].des[j] == false))
        {
          dcopy.setocc(oplist[i].cre[j], false);
          dcopy.setocc(oplist[i].des[j], true);
        }
        else
        {
          valid = false;
          break;
        }
      }

      if (valid) {
        double ovlpdetcopy;
        int excitationDistance = dcopy.ExcitationDistance(walk.getDet());

        if (excitationDistance == 0) {
          ovlpdetcopy = 1.0;
        }
        //if there is a repeat of indices then sometimes a double
        //excitation can only result in excitaiton distance of 1
        if (excitationDistance == 1) {
          int c1 = -1, d1 =-1;
          if (dcopy.getocc(oplist[i].cre[0]) == false)
            c1 = oplist[i].cre[0];
          else
            c1 = oplist[i].cre[1];

          if (dcopy.getocc(oplist[i].des[0]) == true)
            d1 = oplist[i].des[0];
          else
            d1 = oplist[i].des[1];

          double ovlpdetcopy = wave.getJastrowFactor(c1 / 2, d1 / 2, dcopy, walk.d);
          if (c1%2 == 0)
            ovlpdetcopy *= walk.getDetFactorA(c1 / 2, d1 / 2, wave, false);
        }
        else if (excitationDistance == 2) {
          int I = oplist[i].cre[0], J = oplist[i].cre[1], A = oplist[i].des[0], B = oplist[i].des[1]; 
          double JastrowFactor = wave.getJastrowFactor(I / 2, J / 2, A / 2, B / 2, dcopy, walk.d);

          bool doparity = false;
          if (I % 2 == J % 2 && I % 2 == 0)
            ovlpdetcopy = walk.getDetFactorA(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else if (I % 2 == J % 2 && I % 2 == 1)
            ovlpdetcopy = walk.getDetFactorB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else if (I % 2 != J % 2 && I % 2 == 0)
            ovlpdetcopy = walk.getDetFactorAB(I / 2, J / 2, A / 2, B / 2, *this, doparity) * JastrowFactor;
          else
            ovlpdetcopy = walk.getDetFactorAB(J / 2, I / 2, B / 2, A / 2, *this, doparity) * JastrowFactor;
        }
        else {
          cout <<"higher than triple excitation not yet supported."<<endl;
          exit(0);
        }
        gradcopy[i] += ovlpdetcopy;
      }
    }

    double totalOvlp = 0.0;
    for (int i=0; i<grad.rows(); i++) {
      totalOvlp += ciCoeffs[i] * gradcopy[i];
    }
    for (int i=0; i<grad.rows(); i++) {
      grad[i] += gradcopy[i]/totalOvlp;
    }
    return totalOvlp;
  }

  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &ham,
                  vector<double> &ovlpRatio, vector<size_t> &excitation1,
                  vector<size_t> &excitation2, vector<double> &HijElements,
                  int &nExcitations, bool fillExcitations = true)
  {

    int ovlpSize = ovlpRatio.size();
    nExcitations = 0;

    double TINY = schd.screen;
    double THRESH = schd.epsilon;

    Determinant &d = walk.d;

    int norbs = Determinant::norbs;
    vector<int> closed;
    vector<int> open;
    d.getOpenClosed(open, closed);

    //noexcitation
    {
      double E0 = d.Energy(I1, I2, coreE);
      ovlp = Overlap(walk);
      ham = E0;
    }

    //Single alpha-beta excitation
    {
      double time = getTime();
      for (int i = 0; i < closed.size(); i++)
      {
        for (int a = 0; a < open.size(); a++)
        {
          if (closed[i] % 2 == open[a] % 2 && abs(I2hb.Singles(closed[i], open[a])) > THRESH)
          {
            prof.SinglesCount++;

            int I = closed[i] / 2, A = open[a] / 2;
            double tia = 0;
            Determinant dcopy = d;
            bool Alpha = closed[i] % 2 == 0 ? true : false;

            bool doparity = true;
            if (schd.Hamiltonian == HUBBARD)
            {
              tia = I1(2 * A, 2 * I);
              doparity = false;
            }
            else
            {
              tia = I1(2 * A, 2 * I);
              int X = max(I, A), Y = min(I, A);
              int pairIndex = X * (X + 1) / 2 + Y;
              size_t start = I2hb.startingIndicesSingleIntegrals[pairIndex];
              size_t end = I2hb.startingIndicesSingleIntegrals[pairIndex + 1];
              float *integrals = I2hb.singleIntegrals;
              short *orbIndices = I2hb.singleIntegralsPairs;
              for (size_t index = start; index < end; index++)
              {
                if (fabs(integrals[index]) < TINY)
                  break;
                int j = orbIndices[2 * index];
                if (closed[i] % 2 == 1 && j % 2 == 1)
                  j--;
                else if (closed[i] % 2 == 1 && j % 2 == 0)
                  j++;

                if (d.getocc(j))
                {
                  tia += integrals[index];
                }
                //cout << tia<<"  "<<a<<"  "<<integrals[index]<<endl;
              }
              double sgn = 1.0;
              if (Alpha)
                d.parityA(A, I, sgn);
              else
                d.parityB(A, I, sgn);
              tia *= sgn;
            }

            double localham = 0.0;
            if (abs(tia) > THRESH)
            {
              Walker walkcopy = walk;
              walkcopy.exciteWalker(wave, closed[i]*2*norbs+open[a], 0, norbs);
              double ovlpdetcopy = Overlap(walkcopy);
              ham += ovlpdetcopy * tia / ovlp;

              if (fillExcitations)
              {
                if (ovlpSize <= nExcitations)
                {
                  ovlpSize += 1000000;
                  ovlpRatio.resize(ovlpSize);
                  excitation1.resize(ovlpSize);
                  excitation2.resize(ovlpSize);
                  HijElements.resize(ovlpSize);
                }

                ovlpRatio[nExcitations] = ovlpdetcopy;
                excitation1[nExcitations] = closed[i] * 2 * norbs + open[a];
                excitation2[nExcitations] = 0;
                HijElements[nExcitations] = tia;
                nExcitations++;


              }
            }
          }
        }
      }
      prof.SinglesTime += getTime() - time;
    }

    if (schd.Hamiltonian == HUBBARD)
      return;

    //Double excitation
    {
      double time = getTime();

      int nclosed = closed.size();
      for (int ij = 0; ij < nclosed * nclosed; ij++)
      {
        int i = ij / nclosed, j = ij % nclosed;
        if (i <= j)
          continue;
        int I = closed[i] / 2, J = closed[j] / 2;
        int X = max(I, J), Y = min(I, J);

        int pairIndex = X * (X + 1) / 2 + Y;
        size_t start = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
        size_t end = closed[i] % 2 == closed[j] % 2 ? I2hb.startingIndicesSameSpin[pairIndex + 1] : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
        float *integrals = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
        short *orbIndices = closed[i] % 2 == closed[j] % 2 ? I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;

        // for all HCI integrals
        for (size_t index = start; index < end; index++)
        {
          // if we are going below the criterion, break
          if (fabs(integrals[index]) < THRESH)
            break;

          // otherwise: generate the determinant corresponding to the current excitation
          int a = 2 * orbIndices[2 * index] + closed[i] % 2, b = 2 * orbIndices[2 * index + 1] + closed[j] % 2;
          if (!(d.getocc(a) || d.getocc(b)))
          {
            int A = a / 2, B = b / 2;
            double tiajb = integrals[index];

            Walker walkcopy = walk;
            walkcopy.exciteWalker(wave, closed[i] * 2 * norbs + a, closed[j] * 2 * norbs + b, norbs);
            double ovlpdetcopy = Overlap(walkcopy);

            double parity = 1.0;
            if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 0)
              walk.d.parityAA(I, J, A, B, parity); 
            else if (closed[i] % 2 == closed[j] % 2 && closed[i] % 2 == 1)
              walk.d.parityBB(I, J, A, B, parity); 
            else if (closed[i] % 2 != closed[j] % 2 && closed[i] % 2 == 0)
               {walk.d.parityA(I, A, parity) ; walk.d.parityB(J, B, parity);}
            else
               {walk.d.parityB(I, A,parity); walk.d.parityA(J, B, parity);}
            
            ham += ovlpdetcopy * tiajb * parity / ovlp;

            if (fillExcitations)
            {
              if (ovlpSize <= nExcitations)
              {
                ovlpSize += 100000;
                ovlpRatio.resize(ovlpSize);
                excitation1.resize(ovlpSize);
                excitation2.resize(ovlpSize);
                HijElements.resize(ovlpSize);
              }

              ovlpRatio[nExcitations] = ovlpdetcopy;
              excitation1[nExcitations] = closed[i] * 2 * norbs + a;
              excitation2[nExcitations] = closed[j] * 2 * norbs + b;
              HijElements[nExcitations] = tiajb;
              nExcitations++;
            }
          }
        }
      }
      prof.DoubleTime += getTime() - time;
    }
  }

  vector<Determinant> &getDeterminants() { return wave.determinants; }
  vector<double> &getciExpansion() { return wave.ciExpansion; }
  MatrixXd& getHforbsA() {return wave.HforbsA;}
  MatrixXd& getHforbsB() {return wave.HforbsB;}
};

#endif
