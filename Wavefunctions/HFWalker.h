/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef HFWalker_HEADER_H
#define HFWalker_HEADER_H

#include "Determinants.h"
#include "WalkerHelper.h"
#include <array>
#include "igl/slice.h"
#include "igl/slice_into.h"

using namespace Eigen;

/**
 * Is essentially a single determinant used in the VMC/DMC simulation
 * At each step in VMC one need to be able to calculate the following
 * quantities
 * a. The local energy = <walker|H|Psi>/<walker|Psi>
 * b. The gradient     = <walker|H|Psi_t>/<walker/Psi>
 * c. The update       = <walker'|Psi>/<walker|Psi>
 *
 * To calculate these efficiently the walker uses the HFWalkerHelper class
 *
 **/

template<typename Corr, typename Reference> //Corr = CPS/JAstrow or Reference = RHF/UHF/GHF
struct HFWalker
{

  Determinant d;
  WalkerHelper<Corr> cpshelper;
  WalkerHelper<Reference> helper;

  HFWalker() {};
  HFWalker(const Reference &w, const Corr &cps) 
  {
    initDet(w.getHforbsA(), w.getHforbsB());
    helper = WalkerHelper<Reference>(w, d);
    cpshelper = WalkerHelper<Corr>(cps, d);
  }

  HFWalker(const Reference &w, const Corr& cps, const Determinant &pd) : d(pd), helper(w, pd), cpshelper(cps, pd) {}; 

  Determinant& getDet() {return d;}
  void readBestDeterminant(Determinant& d) const 
  {
    if (commrank == 0) {
      char file[5000];
      sprintf(file, "BestDeterminant.txt");
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> d;
    }
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(Determinant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
  {
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;

    d = Determinant();
    for (int i = 0; i < nalpha; i++) {
      int bestorb = 0;
      double maxovlp = 0;
      for (int j = 0; j < norbs; j++) {
        if (abs(HforbsA(i, j)) > maxovlp && !d.getoccA(j)) {
          maxovlp = abs(HforbsA(i, j));
          bestorb = j;
        }
      }
      d.setoccA(bestorb, true);
    }
    for (int i = 0; i < nbeta; i++) {
      int bestorb = 0;
      double maxovlp = 0;
      for (int j = 0; j < norbs; j++) {
        if (schd.hf == "rhf" || schd.hf == "uhf") {
          if (abs(HforbsB(i, j)) > maxovlp && !d.getoccB(j)) {
            bestorb = j;
            maxovlp = abs(HforbsB(i, j));
          }
        }
        else {
          if (abs(HforbsB(i+norbs, j)) > maxovlp && !d.getoccB(j)) {
            bestorb = j;
            maxovlp = abs(HforbsB(i+norbs, j));
          }
        }
      }
      d.setoccB(bestorb, true);
    }
  }

  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
  {
    bool readDeterminant = false;
    char file[5000];
    sprintf(file, "BestDeterminant.txt");

    {
      ifstream ofile(file);
      if (ofile)
        readDeterminant = true;
    }
    if (readDeterminant)
      readBestDeterminant(d);
    else
      guessBestDeterminant(d, HforbsA, HforbsB);
  }

  double getIndividualDetOverlap(int i) const
  {
    return helper.thetaDet[i][0] * helper.thetaDet[i][1];
  }

  double getDetOverlap(const Reference &w) const
  {
    double ovlp = 0.0;
    for (int i = 0; i < helper.thetaDet.size(); i++) {
      ovlp += w.getciExpansion()[i] * helper.thetaDet[i][0] * helper.thetaDet[i][1];
    }
    return ovlp;
  }

  double getDetFactor(int i, int a, const Reference &w) const 
  {
    if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, w);
    else                                   
      return getDetFactor(i / 2, a / 2, 1, w);
  }

  double getDetFactor(int I, int J, int A, int B, const Reference &w) const 
  {
    if (I % 2 == J % 2 && I % 2 == 0)
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 0, w);
    else if (I % 2 == J % 2 && I % 2 == 1)                  
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 1, w);
    else if (I % 2 != J % 2 && I % 2 == 0)                  
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 1, w);
    else                                                    
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 0, w);
  }

  double getDetFactor(int i, int a, bool sz, const Reference &w) const
  {
    int tableIndexi, tableIndexa;
    helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int j = 0; j < w.getDeterminants().size(); j++)
    {
      double factor = helper.rTable[j][sz](tableIndexa, tableIndexi);
      detFactorNum += w.getciExpansion()[j] * factor * helper.thetaDet[j][0] * helper.thetaDet[j][1];
      detFactorDen += w.getciExpansion()[j] * helper.thetaDet[j][0] * helper.thetaDet[j][1];
    }
    return detFactorNum / detFactorDen;
  }

  double getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const Reference &w) const
  {
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    helper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
    helper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;

    double detFactorNum = 0.0;
    double detFactorDen = 0.0;
    for (int j = 0; j < w.getDeterminants().size(); j++)
    {
      double factor;
      if (sz1 == sz2 || helper.hftype == 2)
        factor = helper.rTable[j][sz1](tableIndexa, tableIndexi) * helper.rTable[j][sz1](tableIndexb, tableIndexj) 
            - helper.rTable[j][sz1](tableIndexb, tableIndexi) *helper.rTable[j][sz1](tableIndexa, tableIndexj);
      else
        factor = helper.rTable[j][sz1](tableIndexa, tableIndexi) * helper.rTable[j][sz2](tableIndexb, tableIndexj);
      detFactorNum += w.getciExpansion()[j] * factor * helper.thetaDet[j][0] * helper.thetaDet[j][1];
      detFactorDen += w.getciExpansion()[j] * helper.thetaDet[j][0] * helper.thetaDet[j][1];
    }
    return detFactorNum / detFactorDen;
  }

  void update(int i, int a, bool sz, const Reference &w, const Corr &cps, bool doparity = true)
  {
    double p = 1.0;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (helper.hftype == Generalized) {
      int norbs = Determinant::norbs;
      vector<int> cre{ a + sz * norbs }, des{ i + sz * norbs };
      helper.excitationUpdateGhf(w, cre, des, sz, p, d);
    }
    else
    {
      vector<int> cre{ a }, des{ i };
      helper.excitationUpdate(w, cre, des, sz, p, d);
    }

    cpshelper.updateHelper(cps, d);
  }

  void update(int i, int j, int a, int b, bool sz, const Reference &w, const Corr& cps, bool doparity = true)
  {
    double p = 1.0;
    Determinant dcopy = d;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (doparity) p *= d.parity(b, j, sz);
    d.setocc(j, sz, false);
    d.setocc(b, sz, true);
    if (helper.hftype == Generalized) {
      int norbs = Determinant::norbs;
      vector<int> cre{ a + sz * norbs, b + sz * norbs }, des{ i + sz * norbs, j + sz * norbs };
      helper.excitationUpdateGhf(w, cre, des, sz, p, d);
    }
    else {
      vector<int> cre{ a, b }, des{ i, j };
      helper.excitationUpdate(w, cre, des, sz, p, d);
    }
    cpshelper.updateHelper(cps, d);
  }

  void updateWalker(const Reference& w, const Corr& cps, int ex1, int ex2, bool doparity = true)
  {
    int norbs = Determinant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    if (I % 2 == J % 2 && ex2 != 0) {
      if (I % 2 == 1) {
        update(I / 2, J / 2, A / 2, B / 2, 1, w, cps, doparity);
      }
      else {
        update(I / 2, J / 2, A / 2, B / 2, 0, w, cps, doparity);
      }
    }
    else {
      if (I % 2 == 0)
        update(I / 2, A / 2, 0, w, cps, doparity);
      else
        update(I / 2, A / 2, 1, w, cps, doparity);

      if (ex2 != 0) {
        if (J % 2 == 1) {
          update(J / 2, B / 2, 1, w, cps, doparity);
        }
        else {
          update(J / 2, B / 2, 0, w, cps, doparity);
        }
      }
    }
  }

  void exciteWalker(const Reference& w, const Corr& cps, int excite1, int excite2, int norbs)
  {
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

    if (I1 % 2 == 0)
      update(I1 / 2, A1 / 2, 0, w, cps);
    else
      update(I1 / 2, A1 / 2, 1, w, cps);

    if (excite2 != 0) {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        update(I2 / 2, A2 / 2, 0, w, cps);
      else
        update(I2 / 2, A2 / 2, 1, w, cps);
    }
  }

  void OverlapWithGradient(const Reference &w, Eigen::VectorXd &grad, double detovlp) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;

    //K and L are relative row and col indices
    int KA = 0, KB = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        for (int det = 0; det < w.getDeterminants().size(); det++) {
          Determinant refDet = w.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccA(l)) {
              grad(k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[0](L, KA) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
              L++;
            }
          }
        }
        KA++;
      }
      if (walkerDet.getoccB(k)) {
        for (int det = 0; det < w.getDeterminants().size(); det++) {
          Determinant refDet = w.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccB(l)) {
              if (helper.hftype == UnRestricted)
                grad(norbs * norbs + k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[1](L, KB) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
              else
                grad(k * norbs + l) += w.getciExpansion()[det] * helper.thetaInv[1](L, KB) * helper.thetaDet[det][0] * helper.thetaDet[det][1] / detovlp;
              L++;
            }
          }
        }
        KB++;
      }
    }
  }

  void OverlapWithGradientGhf(const Reference &w, Eigen::VectorXd &grad, double detovlp) const
  {
    int norbs = Determinant::norbs;
    Determinant walkerDet = d;
    Determinant refDet = w.getDeterminants()[0];

    //K and L are relative row and col indices
    int K = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(2 * k * norbs + l) += helper.thetaInv[0](L, K) * helper.thetaDet[0][0] / detovlp;
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(2 * k * norbs + norbs + l) += helper.thetaInv[0](L, K) * helper.thetaDet[0][0] / detovlp;
            //grad(w.getNumJastrowVariables() + w.getciExpansion().size() + k*norbs+l) += walk.alphainv(L, KA);
            L++;
          }
        }
        K++;
      }
    }
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccB(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(2 * norbs * norbs +  2 * k * norbs + l) += helper.thetaDet[0][0] * helper.thetaInv[0](L, K) / detovlp;
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(2 * norbs * norbs +  2 * k * norbs + norbs + l) += helper.thetaDet[0][0] * helper.thetaInv[0](L, K) / detovlp;
            L++;
          }
        }
        K++;
      }
    }
  }

  ostream& operator<<(ostream& os) {
    os << d << endl << endl;
    os << "alphaTable\n" << helper.rTable[0][0] << endl << endl;
    os << "betaTable\n" << helper.rTable[0][1] << endl << endl;
    os << "dets\n" << helper.thetaDet[0][0] << "  " << helper.thetaDet[0][1] << endl << endl;
    os << "alphaInv\n" << helper.thetaInv[0] << endl << endl;
    os << "betaInv\n" << helper.thetaInv[1] << endl << endl;
    return os;
  }
};

#endif
