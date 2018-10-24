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
#ifndef CPSSlater_HEADER_H
#define CPSSlater_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "Slater.h"
#include "CPS.h"
#include "HFWalker.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;
class Determinant;


/**
 * This is the wavefunction, it is a product of the CPS and a linear combination of
 * slater determinants
 */
template<typename Corr, typename Reference>  //Corr = CPS/JAstrow or Reference = RHF/U
struct CPSSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cps
        & slater;
  }

 public:
  using CorrType = Corr;
  using ReferenceType = Reference;
  
  Corr cps; //The jastrow factors
  Reference slater; //reference

  CPSSlater() {};
  double getJastrowFactor(int i, int a, Determinant &dcopy, Determinant &d) const ;
  double getJastrowFactor(int i, int j, int a, int b, Determinant &dcopy, Determinant &d) const;
  Reference& getRef() { return slater; }
  Corr& getCPS() { return cps; }

  void initWalker(HFWalker<Corr, Reference> &walk) const {
    walk = HFWalker<Corr, Reference>(slater, cps);
  }
  
  void initWalker(HFWalker<Corr, Reference> &walk, Determinant &d) const {
    walk = HFWalker<Corr, Reference>(slater, cps, d);
  }
  

  /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
  double Overlap(const HFWalker<Corr, Reference> &walk) const 
  {
    return cps.Overlap(walk.d) * walk.getDetOverlap(slater);
  }


  double getOverlapFactor(const HFWalker<Corr, Reference>& w, Determinant& dcopy, bool doparity=false) const {
    double ovlpdetcopy;
    int excitationDistance = dcopy.ExcitationDistance(walk.d);
    
    if (excitationDistance == 0)
    {
      ovlpdetcopy = 1.0;
    }
    else if (excitationDistance == 1)
    {
      int I, A;
      getDifferenceInOccupation(walk.d, dcopy, I, A);
      ovlpdetcopy = getOverlapFactor(I, A, walk, doparity);
    }
    else if (excitationDistance == 2)
    {
      int I, J, A, B;
      getDifferenceInOccupation(walk.d, dcopy, I, J, A, B);
      ovlpdetcopy = getOverlapFactor(I, J, A, B, walk, doparity);
    }
    else
    {
      cout << "higher than triple excitation not yet supported." << endl;
      exit(0);
    }
    return ovlpdetcopy;
  }

  double getOverlapFactor(int i, int a, const HFWalker<Corr, Reference>& walk, bool doparity) const  {
    Determinant dcopy = walk.d;
    dcopy.setocc(i, false);
    dcopy.setocc(a, true);
      
    return walk.cpshelper.OverlapRatio(i, a, cps, dcopy, walk.d)
        * walk.getDetFactor(i, a, slater);
    //* slater.OverlapRatio(i, a, walk, doparity); 
  }

  double getOverlapFactor(int I, int J, int A, int B, const HFWalker<Corr, Reference>& walk, bool doparity) const  {
    //singleexcitation
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
  
    Determinant dcopy = walk.d;
    dcopy.setocc(I, false);
    dcopy.setocc(J, false);
    dcopy.setocc(A, true);
    dcopy.setocc(B, true);

    return walk.cpshelper.OverlapRatio(I, J, A, B, cps, dcopy, walk.d)
        * walk.getDetFactor(I, J, A, B, slater);
        //* slater.OverlapRatio(I, J, A, B, walk, doparity);
  }



  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
  void OverlapWithGradient(const HFWalker<Corr, Reference> &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    double factor1 = 1.0;
    cps.OverlapWithGradient(walk.d, grad, factor1);

    size_t cpsSize = cps.getNumVariables();
    
    int norbs = Determinant::norbs;
    double detovlp = walk.getDetOverlap(slater);
    for (int k = 0; k < slater.ciExpansion.size(); k++)
      grad[k+cpsSize] += walk.getIndividualDetOverlap(k) / detovlp;
    if (slater.determinants.size() <= 1 && schd.optimizeOrbs) {
      //if (hftype == UnRestricted)
      VectorXd gradOrbitals;
      if (slater.hftype == UnRestricted) {
        gradOrbitals = VectorXd::Zero(2*slater.HforbsA.rows()*slater.HforbsA.rows());
        walk.OverlapWithGradient(slater, gradOrbitals, detovlp);
      }
      else {
        gradOrbitals = VectorXd::Zero(slater.HforbsA.rows()*slater.HforbsA.rows());
        if (slater.hftype == Restricted) walk.OverlapWithGradient(slater, gradOrbitals, detovlp);
        else walk.OverlapWithGradientGhf(slater, gradOrbitals, detovlp);
      }
      for (int i=0; i<gradOrbitals.size(); i++)
        grad[cpsSize + slater.ciExpansion.size() + i] += gradOrbitals[i];
    }

  }

  void printVariables() const
  {
    cps.printVariables();
    slater.printVariables();
  }

  void updateVariables(Eigen::VectorXd &v) 
  {
    cps.updateVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows()-cps.getNumVariables());
    slater.updateVariables(vtail);
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    if (v.rows() != getNumVariables())
      v = VectorXd::Zero(getNumVariables());

    cps.getVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows()-cps.getNumVariables());
    slater.getVariables(vtail);
  }


  long getNumJastrowVariables() const
  {
    return cps.getNumVariables();
  }
  //factor = <psi|w> * prefactor;

  long getNumVariables() const
  {
    int norbs = Determinant::norbs;
    long numVars = 0;
    numVars += getNumJastrowVariables();
    numVars += slater.getNumVariables();

    return numVars;
  }

  string getfileName() const {
    return slater.getfileName()+cps.getfileName();
  }
  
  void writeWave() const
  {
    if (commrank == 0)
    {
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ofstream outfs(file, std::ios::binary);
      boost::archive::binary_oarchive save(outfs);
      save << *this;
      outfs.close();
    }
  }

  void readWave()
  {
    if (commrank == 0)
    {
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ifstream infs(file, std::ios::binary);
      boost::archive::binary_iarchive load(infs);
      load >> *this;
      infs.close();
    }
#ifndef SERIAL
    boost::mpi::communicator world;
    boost::mpi::broadcast(world, *this, 0);
#endif
  }


  //<psi_t| (H-E0) |D>

  void HamAndOvlp(const HFWalker<Corr, Reference> &walk,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true) const
  {
    int norbs = Determinant::norbs;

    ovlp = Overlap(walk);
    ham = walk.d.Energy(I1, I2, coreE); 

    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
  
    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

      ham += tia * ovlpRatio;

      work.ovlpRatio[i] = ovlpRatio;
    }
  }

  void HamAndOvlpLanczos(const HFWalker<Corr, Reference> &walk,
                         Eigen::VectorXd &lanczosCoeffsSample,
                         double &ovlpSample,
                         workingArray& work,
                         workingArray& moreWork, double &alpha)
  {
    work.setCounterToZero();
    int norbs = Determinant::norbs;

    double el0 = 0., el1 = 0., ovlp0 = 0., ovlp1 = 0.;
    HamAndOvlp(walk, ovlp0, el0, work);
    std::vector<double> ovlp{0., 0., 0.};
    ovlp[0] = ovlp0;
    ovlp[1] = el0 * ovlp0;
    ovlp[2] = ovlp[0] + alpha * ovlp[1];

    lanczosCoeffsSample[0] = ovlp[0] * ovlp[0] * el0 / (ovlp[2] * ovlp[2]);
    lanczosCoeffsSample[1] = ovlp[0] * ovlp[1] * el0 / (ovlp[2] * ovlp[2]);
    el1 = walk.d.Energy(I1, I2, coreE);

    //workingArray work1;
    //cout << "E0  " << el1 << endl;
    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      HFWalker<Corr, Reference> walkCopy = walk;
      walkCopy.updateWalker(slater, cps, work.excitation1[i], work.excitation2[i], false);
      moreWork.setCounterToZero();
      HamAndOvlp(walkCopy, ovlp0, el0, moreWork);
      ovlp1 = el0 * ovlp0;
      el1 += tia * ovlp1 / ovlp[1];
      work.ovlpRatio[i] = (ovlp0 + alpha * ovlp1) / ovlp[2];
    }

    lanczosCoeffsSample[2] = ovlp[1] * ovlp[1] * el1 / (ovlp[2] * ovlp[2]);
    lanczosCoeffsSample[3] = ovlp[0] * ovlp[0] / (ovlp[2] * ovlp[2]);
    ovlpSample = ovlp[2];
  }
  
};


#endif
