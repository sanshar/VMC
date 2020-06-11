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
#ifndef relHFWalker_HEADER_H
#define relHFWalker_HEADER_H

#include "Determinants.h"
#include "relDeterminants.h"
#include "WalkerHelper.h"
#include "relWalkerHelper.h"
#include "relCorrelatedWavefunction.h"
#include <array>
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "relSlater.h"
#include "AGP.h"
#include "Pfaffian.h"
#include "relWorkingArray.h"
#include "LocalEnergy.h"
#include "relLocalEnergy.h"

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


template<typename Corr>
struct Walker<Corr, relSlater> {

  relDeterminant d;
  WalkerHelper<Corr> corrHelper;
  WalkerHelper<relSlater> refHelper;

  Walker() {};
  
  Walker(const Corr &corr, const relSlater &ref) 
  {
    initDet(ref.getHforbsA().real(), ref.getHforbsB().real()); //EDIT DO: does walker need complex initialisation ?
    refHelper = WalkerHelper<relSlater>(ref, d);
    corrHelper = WalkerHelper<Corr>(corr, d);
  }

  Walker(const Corr &corr, const relSlater &ref, const relDeterminant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

  relDeterminant& getDet() {return d;}
  void readBestDeterminant(relDeterminant& d) const 
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
  void guessBestDeterminant(relDeterminant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
  {
    cout << "PROBLEM nonrelGuess " <<  nelec << endl;
    int norbs = relDeterminant::norbs;
    int nalpha = relDeterminant::nalpha;
    int nbeta = relDeterminant::nbeta;

    d = relDeterminant();
    if (boost::iequals(schd.determinantFile, "")) {
      for (int i = 0; i < nalpha; i++) {
        int bestorb = 0;
        double maxovlp = 0;
        for (int j = 0; j < norbs; j++) {
          if (abs(HforbsA(j, i)) > maxovlp && !d.getoccA(j)) {
            maxovlp = abs(HforbsA(j, i));
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
            if (abs(HforbsB(j, i)) > maxovlp && !d.getoccB(j)) {
              bestorb = j;
              maxovlp = abs(HforbsB(j, i));
            }
          }
          else {
            if (abs(HforbsB(j+norbs, i+nalpha)) > maxovlp && !d.getoccB(j)) {
              bestorb = j;
              maxovlp = abs(HforbsB(j+norbs, i+nalpha));
            }
          }
        }
        d.setoccB(bestorb, true);
      }
    }
    else if (boost::iequals(schd.determinantFile, "bestDet")) {
      std::vector<relDeterminant> dets;
      std::vector<double> ci;
      readDeterminants(schd.determinantFile, dets, ci);
      d = dets[0];
    }
  }


  void relGuessBestDeterminant(relDeterminant& d, const Eigen::MatrixXd& HforbsA) const //EDIT: no spin restriction for filling initial det
  {
    int norbs = relDeterminant::norbs;
    int nelec = relDeterminant::nalpha + relDeterminant::nbeta;
    //cout << "relGuess " <<  nelec << endl;
    d = relDeterminant();
    if (boost::iequals(schd.determinantFile, "")) {
      for (int i = 0; i < nelec; i++) {
        int bestorb = 0;
        double maxovlp = 0;
        for (int j = 0; j < norbs*2; j++) {
          if (abs(HforbsA(j, i)) > maxovlp && !d.getocc(j)) {
            maxovlp = abs(HforbsA(j, i));
            bestorb = j;
          }
        }
        //cout << bestorb << endl;
        d.setocc(bestorb, true);
      }
    //cout << d << endl;
    }
    else if (boost::iequals(schd.determinantFile, "bestDet")) {
      std::vector<relDeterminant> dets;
      std::vector<double> ci;
      readDeterminants(schd.determinantFile, dets, ci);
      d = dets[0];
    }
  }


  void relMakeHighlyExcitedDeterminant(relDeterminant& d) const //EDIT: no spin restriction for filling initial det
  {
    int norbs = relDeterminant::norbs;
    int nelec = relDeterminant::nalpha + relDeterminant::nbeta;
    cout << "relHighlyExcited " <<  nelec << endl;
    d = relDeterminant();
    for (int i = 0; i < nelec; i++) {
      int bestorb = (norbs*2)-i-1;
      d.setocc(bestorb, true);
    }
    cout << d << endl;
  }


  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
  {
    //cout << "initDet " << schd.hf << endl;
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
    else  //EDIT: different initialisation than other ghf, allows crazy spin states
      relGuessBestDeterminant(d, HforbsA);
      //relMakeHighlyExcitedDeterminant(d);
    //else
      //guessBestDeterminant(d, HforbsA, HforbsB);
  }






  std::complex<double> getIndividualDetOverlap(int i) const
  {
    return (refHelper.thetaDet[i][0] * refHelper.thetaDet[i][1]);
  }

  std::complex<double> getDetOverlap(const relSlater &ref) const
  {
    std::complex<double> ovlp = 0.0 + 0.0i;
    for (int i = 0; i < refHelper.thetaDet.size(); i++) {
      //cout << "ci Exp " << ref.getciExpansion()[i] << " " << refHelper.thetaDet[i][1] << endl;
      ovlp += ref.getciExpansion()[i] * (refHelper.thetaDet[i][0] * refHelper.thetaDet[i][1]);
    }
    return ovlp;
  }

  std::complex<double> getDetFactor(int i, int a, const relSlater &ref) const // EDIT DO: for spin flip 
  {
    //cout << "in getDetFactor: " << i << "  " << a << endl;
    if (i % 2 != a % 2)
      return relGetDetFactor(i, a, ref);
    else if (i % 2 == 0)
      return getDetFactor(i / 2, a / 2, 0, ref);
    else                                   
      return getDetFactor(i / 2, a / 2, 1, ref);
  }

  std::complex<double> getDetFactor(int I, int J, int A, int B, const relSlater &ref) const 
  {
    if (I % 2 == J % 2 && I % 2 == 0)
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 0, ref);
    else if (I % 2 == J % 2 && I % 2 == 1)                  
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 1, ref);
    else if (I % 2 != J % 2 && I % 2 == 0)                  
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 0, 1, ref);
    else                                                    
      return getDetFactor(I / 2, J / 2, A / 2, B / 2, 1, 0, ref);
  }


  std::complex<double> relGetDetFactor(int i, int a, const relSlater &ref) const // EDIT: also for spin flip
  {
    //cout << "here GHF  " << i << "  " << a << "  " << endl;
    int tableIndexi, tableIndexa;
    refHelper.getRelIndicesGHF(i, tableIndexi, a, tableIndexa); // EDIT: computes the indexes differently for spin flip 
    if (1==0 && i==3 && a==2) cout << "here  " << i << "  " << tableIndexi << "  " << a << "  " << tableIndexa << endl;

    std::complex<double> detFactorNum = 0.0 + 0.0i;
    std::complex<double> detFactorDen = 0.0 + 0.0i;
    if (ref.getDeterminants().size() !=1 )
       cout << "Problem, more than one det!" << endl;
    for (int j = 0; j < ref.getDeterminants().size(); j++)
    {
      std::complex<double> factor0 = (refHelper.rTable[j][0](tableIndexa, tableIndexi) * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]) * ref.getciExpansion()[j] /  getDetOverlap(ref);
      detFactorNum += ref.getciExpansion()[j] * factor0 * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
      detFactorDen += ref.getciExpansion()[j] * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
      //cout << "rTab in relGetDetFac " << j << " " << refHelper.rTable[j][0] << tableIndexa << " " << tableIndexi << endl;
      if (1==0 && i==3 && a==2) {
        cout << "relGetDetFactor " << refHelper.rTable[j][0](tableIndexa, tableIndexi) << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
        cout << "relGetDetFactor " << &(refHelper.rTable[j][0](tableIndexa, tableIndexi)) << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
        cout << "relGetDetFactor " << &(refHelper.rTable[j][0]) << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
        for (int i=0; i<(refHelper.rTable[j][0]).size();++i){        
          cout << " i=" << i << " value=" << (refHelper.rTable[j][0]).coeff( i ) << std::endl;
        }
        cout << "\n";
        cout << "relGetDetFactor " << (refHelper.rTable[j][0]).rows() << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
        cout << "relGetDetFactor " << (refHelper.rTable[j][0]).cols() << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
      }
      //if (i==3 && a==2) cout << "relGetDetFactor " << std::end(refHelper.rTable[j][0]) << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
      //if (i==3 && a==2) cout << "relGetDetFactor " << refHelper.rTable[j][0][tableIndexa][tableIndexi] << endl;//<< getDetOverlap(ref) << refHelper.thetaDet[j][0] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
      //cout << "relGetDetFactor " << getDetOverlap(ref) << refHelper.thetaDet[j][1] << " " << i << " " << a << " " << tableIndexi << " " << tableIndexa << endl;
    }
    return detFactorNum / detFactorDen;
  }


  std::complex<double> getDetFactor(int i, int a, bool sz, const relSlater &ref) const
  {
    //cout << "here  " << i << "  " << a << "  " << sz << endl;
    int tableIndexi, tableIndexa;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz); 
    //cout << "here  " << i << "  " << tableIndexi << "  " << a << "  " << tableIndexa << endl;

    std::complex<double> detFactorNum = 0.0 + 0.0i;
    std::complex<double> detFactorDen = 0.0 + 0.0i;
    for (int j = 0; j < ref.getDeterminants().size(); j++)
    {
      std::complex<double> factor = (refHelper.rTable[j][sz](tableIndexa, tableIndexi) * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]) * ref.getciExpansion()[j] /  getDetOverlap(ref);
      detFactorNum += ref.getciExpansion()[j] * factor * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
      detFactorDen += ref.getciExpansion()[j] * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
    }
    return detFactorNum / detFactorDen;
  }

  std::complex<double> getDetFactor(int i, int j, int a, int b, bool sz1, bool sz2, const relSlater &ref) const
  {
    int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
    refHelper.getRelIndices(i, tableIndexi, a, tableIndexa, sz1); 
    refHelper.getRelIndices(j, tableIndexj, b, tableIndexb, sz2) ;

    std::complex<double> detFactorNum = 0.0;
    std::complex<double> detFactorDen = 0.0;
    for (int j = 0; j < ref.getDeterminants().size(); j++)
    {
      std::complex<double> factor;
      if (sz1 == sz2 || refHelper.hftype == 2)
        factor =((refHelper.rTable[j][sz1](tableIndexa, tableIndexi) * refHelper.rTable[j][sz1](tableIndexb, tableIndexj) 
            - refHelper.rTable[j][sz1](tableIndexb, tableIndexi) *refHelper.rTable[j][sz1](tableIndexa, tableIndexj)) * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]) * ref.getciExpansion()[j]/ getDetOverlap(ref);
      else
        factor = (refHelper.rTable[j][sz1](tableIndexa, tableIndexi) * refHelper.rTable[j][sz2](tableIndexb, tableIndexj) * refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]) * ref.getciExpansion()[j]/ getDetOverlap(ref);
      detFactorNum += ref.getciExpansion()[j] * factor * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
      detFactorDen += ref.getciExpansion()[j] * (refHelper.thetaDet[j][0] * refHelper.thetaDet[j][1]);
    }
    return detFactorNum / detFactorDen;
  }

  void update(int i, int a, bool sz, const relSlater &ref, const Corr &corr, bool doparity = true)
  {
    //cout << "update" << endl;
    double p = 1.0;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (refHelper.hftype == Generalized) {
      int norbs = relDeterminant::norbs;
      vector<int> cre{ a + sz * norbs }, des{ i + sz * norbs };
      refHelper.excitationUpdateGhf(ref, cre, des, sz, p, d);
    }
    else
    {
      vector<int> cre{ a }, des{ i };
      refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    }

    corrHelper.updateHelper(corr, d, i, a, sz);
  }


  void relUpdate(int i, int a, const relSlater &ref, const Corr &corr, bool doparity = true) //EDIT: spin-flip updates
  {
    //cout << "relUpdate " << i << " " << a << endl;
    //parSpinFlip *= -1.0;
    double p = 1.0;
    if (doparity) p *= d.relParity(a, i);
    //cout << d << endl;
    d.setocc(i, false);
    d.setocc(a, true);
    //cout << d << endl;
    if (refHelper.hftype == Generalized) {
      int norbs = relDeterminant::norbs;
      vector<int> cre{ (a / 2) + norbs*(a%2)  }, des{ (i / 2) + norbs*(i%2) };
      refHelper.relExcitationUpdateGhf(ref, cre, des, p, d);
    }
    else
    {
      cout << "not implemented yet, only ghf" << endl;
      exit (0);
    }
    corrHelper.relUpdateHelper(corr, d, i, a);
  }


  void update(int i, int j, int a, int b, bool sz, const relSlater &ref, const Corr& corr, bool doparity = true)
  {
    //cout << "update" << endl;
    double p = 1.0;
    relDeterminant dcopy = d;
    if (doparity) p *= d.parity(a, i, sz);
    d.setocc(i, sz, false);
    d.setocc(a, sz, true);
    if (doparity) p *= d.parity(b, j, sz);
    d.setocc(j, sz, false);
    d.setocc(b, sz, true);
    if (refHelper.hftype == Generalized) {
      int norbs = relDeterminant::norbs;
      vector<int> cre{ a + sz * norbs, b + sz * norbs }, des{ i + sz * norbs, j + sz * norbs };
      refHelper.excitationUpdateGhf(ref, cre, des, sz, p, d);
    }
    else {
      vector<int> cre{ a, b }, des{ i, j };
      refHelper.excitationUpdate(ref, cre, des, sz, p, d);
    }
    corrHelper.updateHelper(corr, d, i, j, a, b, sz);
  }

  void updateWalker(const relSlater &ref, const Corr& corr, int ex1, int ex2, bool doparity = true) // EDIT DO: maybe spin flip here
  {
    //cout << "updateWalker" << endl;
    int norbs = relDeterminant::norbs;
    int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
    int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
    //cout << "ex1, ex2: " << ex1 << " " << ex2 << " I, J, A, B: " << I << " " << J << " " << A << " " << B << endl;
    //if (ex2 == 0) cout << "ex1, ex2: " << ex1 << " " << ex2 << " I, J, A, B: " << I << " " << J << " " << A << " " << B << endl;
    //if (I%2 != A%2) cout << "spin flipped!" << endl;
    if (I % 2 == J % 2 && ex2 != 0) {
      if (I % 2 == 1) {
        update(I / 2, J / 2, A / 2, B / 2, 1, ref, corr, doparity);
      }
      else {
        update(I / 2, J / 2, A / 2, B / 2, 0, ref, corr, doparity);
      }
    }
    else if (I%2 != A%2) {
      //cout << "spin flipped! " << d << endl;
      assert (ex2==0);
      //cout << "now to relUpdate" << endl;
      relUpdate(I, A, ref, corr, doparity);
      //cout << "now on det " << d << endl;
    }
    else {
      if (I % 2 == 0)
        update(I / 2, A / 2, 0, ref, corr, doparity);
      else
        update(I / 2, A / 2, 1, ref, corr, doparity);

      if (ex2 != 0) {
        if (J % 2 == 1) {
          update(J / 2, B / 2, 1, ref, corr, doparity);
        }
        else {
          update(J / 2, B / 2, 0, ref, corr, doparity);
        }
      }
    }
  }

  void exciteWalker(const relSlater &ref, const Corr& corr, int excite1, int excite2, int norbs)
  {
    cout << "exciteWalker" << endl;
    int I1 = excite1 / (2 * norbs), A1 = excite1 % (2 * norbs);

    if (I1 % 2 == 0)
      update(I1 / 2, A1 / 2, 0, ref, corr);
    else
      update(I1 / 2, A1 / 2, 1, ref, corr);

    if (excite2 != 0) {
      int I2 = excite2 / (2 * norbs), A2 = excite2 % (2 * norbs);
      if (I2 % 2 == 0)
        update(I2 / 2, A2 / 2, 0, ref, corr);
      else
        update(I2 / 2, A2 / 2, 1, ref, corr);
    }
  }

  void OverlapWithOrbGradient(const relSlater &ref, Eigen::VectorXd &grad, std::complex<double> detovlp) const
  {
    int norbs = relDeterminant::norbs;
    relDeterminant walkerDet = d;

    //K and L are relative row and col indices
    int KA = 0, KB = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        for (int det = 0; det < ref.getDeterminants().size(); det++) {
          relDeterminant refDet = ref.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccA(l)) {
              grad(2 * k * norbs + 2 * l) += (ref.getciExpansion()[det] * (refHelper.thetaInv[0](L, KA) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) /detovlp).real();
              grad(2 * k * norbs + 2 * l + 1) += (ref.getciExpansion()[det] * (- refHelper.thetaInv[0](L, KA) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) /detovlp).imag();
              L++;
            }
          }
        }
        KA++;
      }
      if (walkerDet.getoccB(k)) {
        for (int det = 0; det < ref.getDeterminants().size(); det++) {
          relDeterminant refDet = ref.getDeterminants()[det];
          int L = 0;
          for (int l = 0; l < norbs; l++) {
            if (refDet.getoccB(l)) {
              if (refHelper.hftype == UnRestricted) {
                grad(2 * norbs * norbs + 2 * k * norbs + 2 * l) += (ref.getciExpansion()[det] * (refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) / detovlp).real();
                grad(2 * norbs * norbs + 2 * k * norbs + 2 * l + 1) += (ref.getciExpansion()[det] * (- refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) / detovlp).imag();
              }
              else {
                grad(2 * k * norbs + 2 * l) += (ref.getciExpansion()[det] * (refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) / detovlp).real();
                grad(2 * k * norbs + 2 * l + 1) += (ref.getciExpansion()[det] * (- refHelper.thetaInv[1](L, KB) * refHelper.thetaDet[det][0] * refHelper.thetaDet[det][1]) / detovlp).imag();
              }
              L++;
            }
          }
        }
        KB++;
      }
    }
  }

  void OverlapWithOrbGradientGhf(const relSlater &ref, Eigen::VectorXd &grad, std::complex<double> detovlp) const
  {
    int norbs = relDeterminant::norbs;
    relDeterminant walkerDet = d;
    relDeterminant refDet = ref.getDeterminants()[0];

    //K and exciterelative row and col indices
    int K = 0;
    for (int k = 0; k < norbs; k++) { //walker indices on the row
      if (walkerDet.getoccA(k)) {
        int L = 0;
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccA(l)) {
            grad(4 * k * norbs + 2 * l) += ((refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).real();
            //grad(4 * k * norbs + 2 * l + 1) += ((refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).imag(); // EDIT DO: new formula for imag grad
            grad(4 * k * norbs + 2 * l + 1) += ((- refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).imag();
            L++;
          }
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(4 * k * norbs + 2 * norbs + 2 * l) += ((refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).real();
            //grad(4 * k * norbs + 2 * norbs + 2 * l + 1) += ((refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).imag(); // EDIT DO
            grad(4 * k * norbs + 2 * norbs + 2 * l + 1) += ((- refHelper.thetaInv[0](L, K) * refHelper.thetaDet[0][0]) / detovlp).imag();
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
            grad(4 * norbs * norbs +  4 * k * norbs + 2 * l) += ((refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).real();
            //grad(4 * norbs * norbs +  4 * k * norbs + 2 * l + 1) += ((refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).imag(); // EDIT DO
            grad(4 * norbs * norbs +  4 * k * norbs + 2 * l + 1) += ((- refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).imag();
            L++;
          } 
        }
        for (int l = 0; l < norbs; l++) {
          if (refDet.getoccB(l)) {
            grad(4 * norbs * norbs +  4 * k * norbs + 2 * norbs + 2 * l) += ((refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).real();
            //grad(4 * norbs * norbs +  4 * k * norbs + 2 * norbs + 2 * l + 1) += ((refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).imag(); // EDIT DO
            grad(4 * norbs * norbs +  4 * k * norbs + 2 * norbs + 2 * l + 1) += ((- refHelper.thetaDet[0][0] * refHelper.thetaInv[0](L, K)) / detovlp).imag();
            L++;
          }
        }
        K++;
      }
    }
  }

  void OverlapWithGradient(const relSlater &ref, Eigen::VectorBlock<VectorXd> &grad) const
  {
    std::complex<double> detovlp = getDetOverlap(ref);
    for (int i = 0; i < ref.ciExpansion.size(); i++)
      grad[i] += (getIndividualDetOverlap(i) / detovlp).real();
    if (ref.determinants.size() <= 1 && schd.optimizeOrbs) {
      //if (hftype == UnRestricted)
      VectorXd gradOrbitals;
      if (ref.hftype == UnRestricted) {
        gradOrbitals = VectorXd::Zero(4 * ref.HforbsA.rows() * ref.HforbsA.rows());
        OverlapWithOrbGradient(ref, gradOrbitals, detovlp);
      }
      else {
        gradOrbitals = VectorXd::Zero(2 * ref.HforbsA.rows() * ref.HforbsA.rows());
        if (ref.hftype == Restricted) OverlapWithOrbGradient(ref, gradOrbitals, detovlp);
        else OverlapWithOrbGradientGhf(ref, gradOrbitals, detovlp);
      }
      for (int i = 0; i < gradOrbitals.size(); i++)
        grad[ref.ciExpansion.size() + i] += gradOrbitals[i];
    }
  }

  friend ostream& operator<<(ostream& os, const Walker<Corr, relSlater>& w) {
    os << w.d << endl << endl;
    os << "alphaTable\n" << w.refHelper.rTable[0][0] << endl << endl;
    os << "betaTable\n" << w.refHelper.rTable[0][1] << endl << endl;
    os << "dets\n" << w.refHelper.thetaDet[0][0] << "  " << w.refHelper.thetaDet[0][1] << endl << endl;
    os << "alphaInv\n" << w.refHelper.thetaInv[0] << endl << endl;
    os << "betaInv\n" << w.refHelper.thetaInv[1] << endl << endl;
    return os;
  }

  void OverlapWithLocalEnergyGradient(const Corr &corr, const relSlater &ref, relWorkingArray &work, Eigen::VectorXd &gradEloc) const
  {
    VectorXd v = VectorXd::Zero(corr.getNumVariables() + ref.getNumVariables());
    corr.getVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.getVariables(vtail);
    LocalEnergySolver Solver(d, work);
    double Eloctest = 0.0;
    stan::math::gradient(Solver, v, Eloctest, gradEloc);
    //make sure no nan values persist
    for (int i = 0; i < gradEloc.rows(); i++)
    {
      if (std::isnan(gradEloc(i)))
        gradEloc(i) = 0.0;
    }
    if (schd.debug)
    {
      //cout << Eloc << "\t|\t" << Eloctest << endl;
      cout << Eloctest << endl;

      //below is very expensive and used only for debugging
      Eigen::VectorXd finiteGradEloc = Eigen::VectorXd::Zero(v.size());
      for (int i = 0; i < v.size(); ++i)
      {
        double dt = 0.00001;
        Eigen::VectorXd vdt = v;
        vdt(i) += dt;
        finiteGradEloc(i) = (Solver(vdt) - Eloctest) / dt;
      }
      for (int i = 0; i < v.size(); ++i)
      {
        cout << finiteGradEloc(i) << "\t" << gradEloc(i) << endl;
      }
    }
  }      
};

#endif
