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
#ifndef relCorrelatedWavefunction_HEADER_H
#define relCorrelatedWavefunction_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "Walker.h"
#ifdef Relativistic
#include "relWalker.h"
#endif
#include "workingArray.h"

class oneInt;
class twoInt;

#ifdef Relativistic
class oneIntSOC;
#endif

class twoIntHeatBathSHM;
class workingArray;
class Determinant;


/**
 * This is the wavefunction, it is a product of the CPS and a linear combination of
 * slater determinants
 */
template<typename Corr, typename Reference>  //Corr = CPS/JAstrow or Reference = RHF/U
struct relCorrelatedWavefunction {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & corr
        & ref;
  }

 public:
  using CorrType = Corr;
  using ReferenceType = Reference;
  
  Corr corr; //The jastrow factors
  Reference ref; //reference

  relCorrelatedWavefunction() {};
  Reference& getRef() { return ref; }
  Corr& getCorr() { return corr; }

  void initWalker(Walker<Corr, Reference> &walk) const 
  {
    walk = Walker<Corr, Reference>(corr, ref);
  }
  
  void initWalker(Walker<Corr, Reference> &walk, Determinant &d) const 
  {
    walk = Walker<Corr, Reference>(corr, ref, d);
  }
  
  /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
  std::complex<double> Overlap(const Walker<Corr, Reference> &walk) const 
  {
    return corr.Overlap(walk.d) * walk.getDetOverlap(ref);
  }


  std::complex<double> getOverlapFactor(const Walker<Corr, Reference>& walk, Determinant& dcopy, bool doparity=false) const 
  {
    std::complex<double> ovlpdetcopy;
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

  std::complex<double> getOverlapFactor(int i, int a, const Walker<Corr, Reference>& walk, bool doparity) const  
  {
    Determinant dcopy = walk.d;
    dcopy.setocc(i, false);
    dcopy.setocc(a, true);
      
    return walk.corrHelper.OverlapRatio(i, a, corr, dcopy, walk.d)
        * walk.getDetFactor(i, a, ref);
    //* slater.OverlapRatio(i, a, walk, doparity); 
  }

  std::complex<double> getOverlapFactor(int I, int J, int A, int B, const Walker<Corr, Reference>& walk, bool doparity) const  
  {
    //singleexcitation
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
  
    Determinant dcopy = walk.d;
    dcopy.setocc(I, false);
    dcopy.setocc(J, false);
    dcopy.setocc(A, true);
    dcopy.setocc(B, true);

    return walk.corrHelper.OverlapRatio(I, J, A, B, corr, dcopy, walk.d)
        * walk.getDetFactor(I, J, A, B, ref);
        //* slater.OverlapRatio(I, J, A, B, walk, doparity);
  }


  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
  void OverlapWithGradient(const Walker<Corr, Reference> &walk,
                           double &factor,
                           Eigen::VectorXcd &grad) const
  {
    double factor1 = 1.0;
    corr.OverlapWithGradient(walk.d, grad, factor1);
  
    Eigen::VectorBlock<VectorXcd> gradtail = grad.tail(grad.rows() - corr.getNumVariables());
    walk.OverlapWithGradient(ref, gradtail);
  }

  void printVariables() const
  {
    corr.printVariables();
    ref.printVariables();
  }

  void updateVariables(Eigen::VectorXcd &v) 
  {
    if (schd.optimizeCps == true)
      corr.updateVariables(v);
    //Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    //ref.updateVariables(vtail);
    if (schd.optimizeOrbs == true)
      ref.updateVariables(v.tail(v.rows() - corr.getNumVariables()));
  }

  void getVariables(Eigen::VectorXcd &v) const
  {
    if (v.rows() != getNumVariables())
      v = VectorXcd::Zero(getNumVariables());

    corr.getVariables(v);
    Eigen::VectorBlock<VectorXcd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.getVariables(vtail);
  }


  long getNumJastrowVariables() const
  {
    return corr.getNumVariables();
  }
  //factor = <psi|w> * prefactor;

  long getNumVariables() const
  {
    int norbs = Determinant::norbs;
    long numVars = 0;
    numVars += getNumJastrowVariables();
    numVars += ref.getNumVariables();

    return numVars;
  }

  string getfileName() const {
    return ref.getfileName() + corr.getfileName();
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

  void HamAndOvlp(const Walker<Corr, Reference> &walk,
                  std::complex<double> &ovlp, double &ham, 
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
    //cout << "eloc excitations" << endl;
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      std::complex<double> ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

      ham += tia * ovlpRatio;
      if (schd.debug) cout << ex1 << "  " << ex2 << "  tia  " << tia << "  ovlpRatio  " << ovlpRatio << endl;

      work.ovlpRatio[i] = ovlpRatio;
    }
  }

  void OverlapWithLocalEnergyGradient(const Walker<Corr, Reference> &walk, workingArray &work, Eigen::VectorXd &gradEloc) const
  {
    walk.OverlapWithLocalEnergyGradient(corr, ref, work, gradEloc);
    int numCpsVars = corr.getNumVariables();
    int numRefVars = ref.getNumVariables();
    int numVars = getNumVariables();
    if (schd.optimizeCps == false)
      gradEloc.head(numCpsVars).setZero();
    if (schd.optimizeOrbs == false)
      gradEloc.tail(numRefVars).setZero();
    if (schd.ifComplex == false)
    {
      for (int i = 0; i < numRefVars; i++)
      {
        if (i % 2 == 1)
          gradEloc(i + numCpsVars) = 0.0;
      }
    }
  }  

  void HamAndOvlpLanczos(const Walker<Corr, Reference> &walk,
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
      Walker<Corr, Reference> walkCopy = walk;
      walkCopy.updateWalker(ref, corr, work.excitation1[i], work.excitation2[i], false);
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