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


#include "rWalkerHelper.h"

rWalkerHelper<rSlater>::rWalkerHelper(const rSlater &w, const rDeterminant &d) 
{
  hftype = w.hftype;
 
  //fill the spin strings for the walker and the zeroth reference det
  thetaDet.resize(w.getNumOfDets());

  if (hftype == Generalized) {
    initInvDetsTablesGhf(w, d);
  }
  else {
    initInvDetsTables(w, d);
  }
}

void rWalkerHelper<rSlater>::initInvDetsTables(const rSlater& w, const rDeterminant &d) {
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);

  DetMatrix[0] = MatrixXd::Zero(d.nalpha, d.nalpha);
  DetMatrix[1] = MatrixXd::Zero(d.nbeta, d.nbeta);

  Gradient.resize(d.nalpha, MatrixXd::Zero(3, d.nalpha));
  Gradient.resize(d.nalpha+d.nbeta, MatrixXd::Zero(3, d.nbeta));
  Laplacian[0] = MatrixXd::Zero(d.nalpha, d.nalpha);

  for (int elec=0; elec<d.nalpha; elec++) {

    schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

    for (int mo=0; mo<d.nalpha; mo++) 
      for (int j=0; j<norbs; j++) {
        DetMatrix[0](elec, mo) += aoValues[j] * w.getHforbs(0)(j, mo);

        Laplacian[0](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] ) * w.getHforbs(0)(j,mo);

        Gradient[elec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(0)(j, mo);
        Gradient[elec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(0)(j, mo);
        Gradient[elec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(0)(j, mo);
      }          
  }
    
  //cout << w.getHforbs(0).col(0)<<endl;
  Eigen::FullPivLU<MatrixXd> lua(DetMatrix[0]);
  if (lua.isInvertible()) {
    thetaInv[0] = lua.inverse();
    thetaDet[0][0] = lua.determinant();
  }
  else {
    cout << " overlap with alpha determinant not invertible" << endl;
    exit(0);
  }

    
  Laplacian[1] = MatrixXd::Zero(d.nbeta, d.nbeta);
  for (int elec=0; elec<d.nbeta; elec++) {
    schd.basis->eval_deriv2(d.coord[elec+d.nalpha], &aoValues[0]);
    for (int mo=0; mo<d.nbeta; mo++) 
      for (int j=0; j<norbs; j++) {
        DetMatrix[1](elec, mo) += aoValues[j] * w.getHforbs(1)(j, mo);

        Laplacian[1](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] ) * w.getHforbs(1)(j,mo);
      
        Gradient[d.nalpha + elec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(1)(j, mo);
        Gradient[d.nalpha + elec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(1)(j, mo);
        Gradient[d.nalpha + elec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(1)(j, mo);
      }
  }

  if (d.nbeta != 0) {
    Eigen::FullPivLU<MatrixXd> lub(DetMatrix[1]);
    if (lub.isInvertible()) {
      thetaInv[1] = lub.inverse();
      thetaDet[0][1] = lub.determinant();
    }
    else {
      cout << " overlap with beta determinant not invertible" << endl;
      exit(0);
    }
  }

    
}


void rWalkerHelper<rSlater>::initInvDetsTablesGhf(const rSlater& w, const rDeterminant &d) {
  int norbs = Determinant::norbs;
  aoValues.resize(10*norbs, 0.0);

  DetMatrix[0] = MatrixXd::Zero(d.nelec, d.nelec);
  Gradient.resize(d.nelec, MatrixXd::Zero(3, d.nelec));
  Laplacian[0] = MatrixXd::Zero(d.nelec, d.nelec);

  AOLaplacian = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[0]  = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[1]  = MatrixXd::Zero(d.nelec, 2*norbs);
  AOGradient[2]  = MatrixXd::Zero(d.nelec, 2*norbs);
    
  for (int elec=0; elec<d.nelec; elec++) {
    schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

    for (int j=0; j<norbs; j++) {
      int J = elec < d.nalpha ? j : j+norbs;
      AOGradient[0](elec, J) = aoValues[1*norbs+j];
      AOGradient[1](elec, J) = aoValues[2*norbs+j];
      AOGradient[2](elec, J) = aoValues[3*norbs+j];
      AOLaplacian  (elec, J) = aoValues[4*norbs+j]
          + aoValues[7*norbs+j]
          + aoValues[9*norbs+j] ;
    }
    
    for (int mo=0; mo<d.nelec; mo++) 
      for (int j=0; j<norbs; j++) {
        int J = elec < d.nalpha ? j : j+norbs;
        DetMatrix[0](elec, mo) += aoValues[j]
                       * (w.getHforbs(0)(J, mo));

        Laplacian[0](elec, mo) += (  aoValues[4*norbs+j]
                                     + aoValues[7*norbs+j]
                                     + aoValues[9*norbs+j] )
                       * (w.getHforbs(0)(J,mo));

        Gradient[elec](0, mo) += aoValues[1*norbs+j]
                       * (w.getHforbs(0)(J,mo));
        Gradient[elec](1, mo) += aoValues[2*norbs+j]
                       * (w.getHforbs(0)(J,mo));
        Gradient[elec](2, mo) += aoValues[3*norbs+j]
                       * (w.getHforbs(0)(J,mo));
          
      }
  }

  Eigen::FullPivLU<MatrixXd> lu(DetMatrix[0]);
  if (lu.isInvertible()) {
    thetaInv[0] = lu.inverse();
    thetaDet[0][0] = lu.determinant();
  }
  else {
    cout << " overlap with GHF determinant not invertible" << endl;
    cout << w.getHforbs(0)<<endl;
    cout << DetMatrix[0].rows()<<"  "<<DetMatrix[0].cols()<<endl;
    cout << DetMatrix[0]<<endl<<endl;
    cout << thetaInv[0]<<endl<<endl;
    exit(0);
  }


}

double rWalkerHelper<rSlater>::getDetFactor(int i, Vector3d& newCoord, const rDeterminant &d,
                                           const rSlater& w) {
  if (hftype == Generalized) 
    return getDetFactorGHF(i, newCoord, 0, d.nelec, w);
  else if (i < d.nalpha)
    return getDetFactor(i, newCoord, 0, d.nalpha, w);
  else
    return getDetFactor(i-d.nalpha, newCoord, 1, d.nbeta, w);
}

double rWalkerHelper<rSlater>::getDetFactorGHF(int i, Vector3d& newCoord,
                                              int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(norbs);

  schd.basis->eval(newCoord, &aoValues[0]);


  VectorXd newVec = VectorXd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) {
      int J = i < rDeterminant::nalpha ? j : j+norbs;
      newVec(mo) += aoValues[j] * (w.getHforbs(sz)(J, mo));
    }
    
  return (newVec.dot(thetaInv[sz].col(i)));
}
  
double rWalkerHelper<rSlater>::getDetFactor(int i, Vector3d& newCoord,
                                           int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(norbs);

  schd.basis->eval(newCoord, &aoValues[0]);


  VectorXd newVec = VectorXd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) 
      newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);

  return (newVec.dot(thetaInv[sz].col(i)));
}

void rWalkerHelper<rSlater>::updateWalker(int i, Vector3d& oldCoord, const rDeterminant &d,
                                         const rSlater& w) {
  if (hftype == Generalized) 
    updateWalkerGHF(i, oldCoord, d, 0, d.nelec, w);
  else if (i < d.nalpha)
    updateWalker(i, oldCoord, d, 0, d.nalpha, w);
  else
    updateWalker(i-d.nalpha, oldCoord, d, 1, d.nbeta, w);

}


void rWalkerHelper<rSlater>::updateWalkerGHF(int elec, Vector3d& oldCoord, const rDeterminant &d,
                                            int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(10 * norbs, 0.0);

  schd.basis->eval_deriv2(d.coord[elec], &aoValues[0]);

  for (int j=0; j<norbs; j++) {
      int J = elec < d.nalpha ? j : j+norbs;
      AOGradient[0](elec, J) = aoValues[1*norbs+j];
      AOGradient[1](elec, J) = aoValues[2*norbs+j];
      AOGradient[2](elec, J) = aoValues[3*norbs+j];
      AOLaplacian  (elec, J) = aoValues[4*norbs+j]
          + aoValues[7*norbs+j]
          + aoValues[9*norbs+j] ;
  }
  
  VectorXd newVec = VectorXd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++) {
      int J = elec < rDeterminant::nalpha ? j : j+norbs;
      newVec(mo) += aoValues[j] * (w.getHforbs(sz)(J, mo));
    }
    
  calculateInverseDeterminantWithRowChange(thetaInv[sz],thetaDet[0][sz],DetMatrix[sz],
                                           elec, newVec);

  for (int mo=0; mo<nelec; mo++) {
    Laplacian[sz](elec, mo) = 0.0;


    Gradient[elec](0, mo) = 0.0;
    Gradient[elec](1, mo) = 0.0;
    Gradient[elec](2, mo) = 0.0;
      
    for (int j=0; j<norbs; j++) {
      int J = elec < rDeterminant::nalpha ? j : j+norbs;
      Laplacian[sz](elec, mo) += (  aoValues[4*norbs+j]
                                    + aoValues[7*norbs+j]
                                    + aoValues[9*norbs+j] )
                     * (w.getHforbs(sz)(J, mo));
        
        
      Gradient[elec](0, mo) += aoValues[1*norbs+j]
                     * (w.getHforbs(sz)(J, mo));
      Gradient[elec](1, mo) += aoValues[2*norbs+j]
                     * (w.getHforbs(sz)(J, mo));
      Gradient[elec](2, mo) += aoValues[3*norbs+j]
                     * (w.getHforbs(sz)(J, mo));
    }
      
  }

}
  
  
void rWalkerHelper<rSlater>::updateWalker(int elec, Vector3d& oldCoord, const rDeterminant &d,
                                         int sz, int nelec, const rSlater& w) {
  int norbs = Determinant::norbs;
  aoValues.resize(10 * norbs, 0.0);

  int gelec = elec;
  if (sz == 1) gelec += d.nalpha;    

  schd.basis->eval_deriv2(d.coord[gelec], &aoValues[0]);

  VectorXd newVec = VectorXd::Zero(nelec);
  for (int mo=0; mo<nelec; mo++) 
    for (int j=0; j<norbs; j++)  
      newVec(mo) += aoValues[j] * w.getHforbs(sz)(j, mo);

    
  calculateInverseDeterminantWithRowChange(thetaInv[sz],thetaDet[0][sz],DetMatrix[sz],
                                           elec, newVec);

  for (int mo=0; mo<nelec; mo++) {
    Laplacian[sz](elec, mo) = 0.0;


    Gradient[gelec](0, mo) = 0.0;
    Gradient[gelec](1, mo) = 0.0;
    Gradient[gelec](2, mo) = 0.0;
      
    for (int j=0; j<norbs; j++) {
      Laplacian[sz](elec, mo) += (  aoValues[4*norbs+j]
                                    + aoValues[7*norbs+j]
                                    + aoValues[9*norbs+j] ) * w.getHforbs(sz)(j,mo);
        
        
      Gradient[gelec](0, mo) += aoValues[1*norbs+j] * w.getHforbs(sz)(j, mo);
      Gradient[gelec](1, mo) += aoValues[2*norbs+j] * w.getHforbs(sz)(j, mo);
      Gradient[gelec](2, mo) += aoValues[3*norbs+j] * w.getHforbs(sz)(j, mo);
    }
      
  }

}


void rWalkerHelper<rSlater>::OverlapWithGradient(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad,
                                                const double& ovlp) 
{
  if (schd.hf == "ghf")
    OverlapWithGradientGhf(d, ref, grad);
  else
    OverlapWithGradient(d, ref, grad);
}

void rWalkerHelper<rSlater>::HamOverlap(const rDeterminant& d, 
                                        const rSlater& ref,
                                        MatrixXd& Rij,
                                        MatrixXd& RiN,
                                        Eigen::VectorBlock<VectorXd>& hamgrad)
{

  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

  const MatrixXd& Bij = Laplacian[0]; //i = nelec , j = norbs
  double kineticTot = 0.0;
  for (int i=0; i<nelec; i++) {
    kineticTot += thetaInv[0].row(i).dot(Bij.col(i));
  }

  double potentialij = 0.0, potentiali=0;

  //get potential
  for (int i=0; i<nelec; i++)
    for (int j=i+1; j<nelec; j++) {
      potentialij += 1./Rij(i,j);
    }
  for (int i=0; i<nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/RiN(i,j);
    }
  }

  MatrixXd AoRi = MatrixXd::Zero(nelec, 2*norbs);
  aoValues.resize(norbs);  
  for (int elec=0; elec<nelec; elec++) {
    schd.basis->eval(d.coord[elec], &aoValues[0]);
    for (int orb = 0; orb<norbs; orb++) 
      if (elec < nalpha)
        AoRi(elec, orb) = aoValues[orb];
      else
        AoRi(elec, norbs+orb) = aoValues[orb];
  }


  //precalculate A-1 B A-1
  MatrixXd X = thetaInv[0] * Laplacian[0] * thetaInv[0];
  
  for (int mo = 0; mo <nelec; mo++) {

    for (int orb = 0; orb < 2*norbs; orb++) {
      double t1 = thetaInv[0].row(mo).dot(AOLaplacian.col(orb));
      double t2 = X          .row(mo).dot(AoRi       .col(orb));
      double t3 = thetaInv[0].row(mo).dot(AoRi       .col(orb));
      
      hamgrad[numDets + orb*nelec + mo] += -0.5*(t1 - t2)
          + (-0.5*kineticTot+potentialij+potentiali) * t3;

      //hamgrad[numDets + (orb+norbs)*nelec + mo] += -0.5*(t1 - t2)
      //+ (-0.5*kineticTot+potentialij+potentiali) * t3;
      
    }
    
  }

  /*
  //*************CHECK**********
  MatrixXd Detder = DetMatrix[0];
  //mo = 1 and ao = 2
  Detder.block(0,1,nelec,1) = AOLaplacian.col(2);
  cout << DetMatrix[0]<<endl<<endl;
  cout << Detder<<endl<<endl;
  cout << thetaInv[0]*DetMatrix[0]<<endl<<endl;
  
  cout << Detder.determinant()/DetMatrix[0].determinant()<<"  "<<thetaInv[0].row(1).dot(AOLaplacian.col(2))<<endl;
  exit(0);
  */
}


void rWalkerHelper<rSlater>::OverlapWithGradient(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad)
{
  grad[0] = 0.0;
  
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  
  MatrixXd AoRia = MatrixXd::Zero(nalpha, norbs);
  MatrixXd AoRib = MatrixXd::Zero(nbeta, norbs);
  aoValues.resize(norbs);
  int numDets = ref.determinants.size();
  
  for (int elec=0; elec<nalpha; elec++) {
    schd.basis->eval(d.coord[elec], &aoValues[0]);
    for (int orb = 0; orb<norbs; orb++)
      AoRia(elec, orb) = aoValues[orb];
  }
  
  for (int elec=0; elec<nbeta; elec++) {
    schd.basis->eval(d.coord[elec+nalpha], &aoValues[0]);
    for (int orb = 0; orb<norbs; orb++)
      AoRib(elec, orb) = aoValues[orb];
  }
  
  //Assuming a single determinant
  for (int moa=0; moa<nalpha; moa++) {//alpha mo 
    for (int orb=0; orb<norbs; orb++) {//ao
      grad[numDets + orb * nalpha + moa] += thetaInv[0].row(moa).dot(AoRia.col(orb));
    }
  }
  
  for (int mob=0; mob<nbeta; mob++) {//beta mo 
    for (int orb=0; orb<norbs; orb++) {//ao
      if (ref.hftype == Restricted) {
        grad[numDets + orb * nbeta + mob] += thetaInv[1].row(mob).dot(AoRib.col(orb));
      }
      else
        grad[numDets + nalpha*norbs + orb * nbeta + mob] += thetaInv[1].row(mob).dot(AoRib.col(orb));
    }
  }
  

}


void rWalkerHelper<rSlater>::OverlapWithGradientGhf(const rDeterminant& d, 
                                                const rSlater& ref,
                                                Eigen::VectorBlock<VectorXd>& grad) 
{
  grad[0] = 0.0;
  
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();
  
  MatrixXd AoRi = MatrixXd::Zero(nelec, 2*norbs);
  aoValues.resize(norbs);
  
  for (int elec=0; elec<nelec; elec++) {
    schd.basis->eval(d.coord[elec], &aoValues[0]);
    for (int orb = 0; orb<norbs; orb++) {
      if (elec < nalpha)
        AoRi(elec, orb) = aoValues[orb];
      else
        AoRi(elec, norbs+orb) = aoValues[orb];
    }
  }

  for (int mo=0; mo<nelec; mo++) {
    for (int orb=0; orb< 2*norbs; orb++) {
      grad[numDets + orb * nelec + mo] += thetaInv[0].row(mo).dot(AoRi.col(orb));
    }
  }

}


rWalkerHelper<rJastrow>::rWalkerHelper(const rJastrow& cps, const rDeterminant& d,
                                       MatrixXd& Rij, MatrixXd& RiN) {

  //RIJ matrix
  //make exponential
  GradRatio = MatrixXd::Zero(d.nelec,3);
  LaplaceRatio = VectorXd::Zero(d.nelec);
  LaplaceRatioIntermediate = VectorXd::Zero(d.nelec);

  rJastrow& j = const_cast<rJastrow&>(cps);
  
  exponential = 0.0;
  exponential = j.exponentialInitLaplaceGrad(d, GradRatio,
                                             LaplaceRatioIntermediate);

  for (int i=0; i<d.nelec; i++) {
    LaplaceRatio[i] = LaplaceRatioIntermediate[i] +
        pow(GradRatio(i,0), 2) +
        pow(GradRatio(i,1), 2) +
        pow(GradRatio(i,2), 2) ;
  }
}

void rWalkerHelper<rJastrow>::updateWalker(int i, Vector3d& oldcoord,
                                           const rJastrow& cps,
                                           const rDeterminant& d,
                                           MatrixXd& Rij, MatrixXd& RiN) {
  cps.UpdateLaplaceGrad(GradRatio, LaplaceRatioIntermediate,
                        Rij, RiN, d, oldcoord, i);

  for (int i=0; i<d.nelec; i++) {
    LaplaceRatio[i] = LaplaceRatioIntermediate[i] +
        pow(GradRatio(i,0), 2) +
        pow(GradRatio(i,1), 2) +
        pow(GradRatio(i,2), 2) ;
  }
}


//the position of the ith electron has changed
double rWalkerHelper<rJastrow>::OverlapRatio(int i, Vector3d& coord, const rJastrow& cps,
                                             const rDeterminant &d) const
{
  return exp(const_cast<rJastrow&>(cps).exponentDiff(i, coord, d));
}


void rWalkerHelper<rJastrow>::OverlapWithGradient(const rJastrow& cps,
                                                  VectorXd& grad,
                                                  const rDeterminant& d,
                                                  const double& ovlp) const {
                                                  
    
  if (schd.optimizeCps) {
    cps.OverlapWithGradient(grad, d);
  }
}

void rWalkerHelper<rJastrow>::HamOverlap(const rJastrow& cps,
                                         VectorXd& grad,
                                         const rDeterminant& d,
                                         const double& ovlp) const {
                                                  
  return;
  //if (schd.optimizeCps) {
  //cps.OverlapWithGradient(grad, d);
  //}
}
