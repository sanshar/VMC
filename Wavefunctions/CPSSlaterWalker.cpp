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
#include "CPSSlaterWalker.h"
#include "CPSSlater.h"
#include "integral.h"
#include "global.h"
#include "input.h"
#include <algorithm>
#include "igl/slice.h"
#include "igl/slice_into.h"
using namespace Eigen;

bool CPSSlaterWalker::operator<(const CPSSlaterWalker& w) const {
  return d < w.d;
}

bool CPSSlaterWalker::operator==(const CPSSlaterWalker& w) const {
  return d == w.d;
}

void CPSSlaterWalker::updateWalker(CPSSlater& w, int ex1, int ex2) {
  int norbs = Determinant::norbs ;

  int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
  int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

  if (I % 2 == J % 2 && ex2 != 0)
    {
      if (I % 2 == 1) {
	updateB(I / 2, J / 2, A / 2, B / 2, w);
      }

      else {
	updateA(I / 2, J / 2, A / 2, B / 2, w);
      }
    }
  else
    {
      if (I % 2 == 0)
	updateA(I / 2, A / 2, w);
      else
	updateB(I / 2, A / 2, w);
      
      if (ex2 != 0)
        {
          if (J % 2 == 1)
	    {
	      updateB(J / 2, B / 2, w);
	    }
          else
	    {
	      updateA(J / 2, B / 2, w);
	    }
	}
    }
  
}

void CPSSlaterWalker::OverlapWithGradient(CPSSlater& w, VectorXd& grad, double detovlp) {
  int numJastrowVariables = w.getNumJastrowVariables();
  int norbs = Determinant::norbs;
  CPSSlaterWalker& walk = *this;

  int KA = 0, KB = 0;
  for (int k = 0; k < norbs; k++) {//walker indices on the row
    if (walk.d.getoccA(k)) {
      
      for (int det = 0; det<w.determinants.size(); det++) {
	Determinant ddet = w.determinants[det];
	int L = 0;
	for (int l = 0; l < norbs; l++) {
	  if (ddet.getoccA(l)) {
	    grad(numJastrowVariables + w.ciExpansion.size() + k*norbs+l) += w.ciExpansion[det]*walk.alphainv(L, KA)*walk.alphaDet[det]*walk.betaDet[det]/detovlp;
	    //grad(w.getNumJastrowVariables() + w.ciExpansion.size() + k*norbs+l) += walk.alphainv(L, KA);
	    L++;
	  }
	}
      }
      KA++;
    }
    if (walk.d.getoccB(k)) {
      
      for (int det = 0; det<w.determinants.size(); det++) {
	Determinant ddet = w.determinants[det];
	int L = 0;
	for (int l = 0; l < norbs; l++) {
	  if (ddet.getoccB(l)) {
	    if (schd.uhf) 
	      grad(numJastrowVariables + w.ciExpansion.size() + norbs*norbs + k*norbs+l) += w.ciExpansion[det]*walk.alphaDet[det]*walk.betaDet[det]*walk.betainv(L, KB)/detovlp;
	    else
	      grad(numJastrowVariables + w.ciExpansion.size() + k*norbs+l) += w.ciExpansion[det]*walk.alphaDet[det]*walk.betaDet[det]*walk.betainv(L, KB)/detovlp;
	    //grad(w.getNumJastrowVariables() + w.ciExpansion.size() + k*norbs+l) += walk.betainv(L, KB);
	    L++;
	  }
	}
      }
      KB++;
    }
  }
  
  
}


bool CPSSlaterWalker::makeMove(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int norbs = Determinant::norbs,
      nalpha = Determinant::nalpha,
      nbeta = Determinant::nbeta;

  //pick a random occupied orbital
  int i = floor(random() * (nalpha + nbeta));
  if (i < nalpha)
  {
    int a = floor(random() * (norbs - nalpha));
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];
    double detfactor = getDetFactorA(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if (pow(detfactor, 2) > random())
    {
      updateA(I, A, w);
      return true;
    }
  }
  else
  {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if ( pow(detfactor, 2) > random() ) {
      updateB(I, A, w);
      return true;
    }

  }

  return false;
}


bool CPSSlaterWalker::makeMovePropPsi(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  int norbs = Determinant::norbs,
    nalpha = Determinant::nalpha,
    nbeta  = Determinant::nbeta;

  //pick a random occupied orbital
  int i = floor( random()*(nalpha+nbeta) );
  if (i < nalpha) {
    int a = floor(random()* (norbs-nalpha) );
    int I = AlphaClosed[i];
    int A = AlphaOpen[a];
    double detfactor = getDetFactorA(I, A, w);

    if ( abs(detfactor) > random() ) {
      updateA(I, A, w);
      return true;
    }

  }
  else {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    int I = BetaClosed[i];
    int A = BetaOpen[a];
    double detfactor = getDetFactorB(I, A, w);

    if ( abs(detfactor) > random() ) {
      updateB(I, A, w);
      return true;
    }

  }

  return false;
}

double CPSSlaterWalker::getDetOverlap(CPSSlater &w)
{
  double ovlp = 0.0;
  for (int i = 0; i < alphaDet.size(); i++) {
    ovlp += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return ovlp;
}

void CPSSlaterWalker::calculateInverseDeterminantWithColumnChange(MatrixXd &inverseIn, double &detValueIn,
                                                         MatrixXd &inverseOut, double &detValueOut,
                                                         vector<int> &cre, vector<int> &des,
                                                         Eigen::Map<Eigen::VectorXi> &RowVec,
                                                         vector<int> &ColIn, MatrixXd &Hforbs)
{
  int ncre = 0, ndes = 0;
  for (int i=0; i<cre.size(); i++)
    if (cre[i] != -1) ncre++;
  for (int i=0; i<des.size(); i++)
    if (des[i] != -1) ndes++;
  if (ncre == 0) {
    inverseOut = inverseIn;
    detValueOut = detValueIn;
    return;
  }


  Eigen::Map<VectorXi> ColCre(&cre[0], ncre); 
  Eigen::Map<VectorXi> ColDes(&des[0], ndes); 

  MatrixXd newCol, oldCol;
  igl::slice(Hforbs, RowVec, ColCre, newCol);
  igl::slice(Hforbs, RowVec, ColDes, oldCol);
  newCol = newCol - oldCol;


  MatrixXd vT = MatrixXd::Zero(ncre, ColIn.size());
  vector<int> ColOutWrong = ColIn;
  for (int i=0; i<ndes; i++) {
    int index = std::lower_bound(ColIn.begin(), ColIn.end(), des[i]) - ColIn.begin();
    vT(i, index) = 1.0;
    ColOutWrong[index] = cre[i];
  }

  //igl::slice(inverseIn, ColCre, 1, vTinverseIn);
  MatrixXd vTinverseIn = vT*inverseIn; 

  MatrixXd Id = MatrixXd::Identity(ncre, ncre);
  MatrixXd detFactor = Id + vTinverseIn*newCol;
  MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<MatrixXd> lub(detFactor);
  if (lub.isInvertible() ) {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseIn * newCol) * detFactorInv) * (vTinverseIn); 
    detValueOut = detValueIn * detFactor.determinant();
  }
  else {
    MatrixXd originalOrbs;
    Eigen::Map<VectorXi> Col(&ColIn[0], ColIn.size());
    igl::slice(Hforbs, RowVec, Col, originalOrbs);
    MatrixXd newOrbs = originalOrbs + newCol*vT; 
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(ColOutWrong.size()), ccopy = ColOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&ccopy](size_t i1, size_t i2) { return ccopy[i1] < ccopy[i2]; });
  Eigen::Map<VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, orderVec, 1, inverseOut);
}

void CPSSlaterWalker::calculateInverseDeterminantWithRowChange(MatrixXd &inverseIn, double &detValueIn,
                                                      MatrixXd &inverseOut, double &detValueOut,
                                                      vector<int> &cre, vector<int> &des,
                                                      Eigen::Map<Eigen::VectorXi> &ColVec,
                                                      vector<int> &RowIn, MatrixXd &Hforbs)
{
  int ncre = 0, ndes = 0;
  for (int i=0; i<cre.size(); i++)
    if (cre[i] != -1) ncre++;
  for (int i=0; i<des.size(); i++)
    if (des[i] != -1) ndes++;
  if (ncre == 0) {
    inverseOut = inverseIn;
    detValueOut = detValueIn;
    return;
  }

  Eigen::Map<VectorXi> RowCre(&cre[0], ncre); 
  Eigen::Map<VectorXi> RowDes(&des[0], ndes); 

  MatrixXd newRow, oldRow;
  igl::slice(Hforbs, RowCre, ColVec, newRow);
  igl::slice(Hforbs, RowDes, ColVec, oldRow);
  newRow = newRow - oldRow;


  MatrixXd U = MatrixXd::Zero(ColVec.rows(), ncre);
  vector<int> RowOutWrong = RowIn;
  for (int i=0; i<ndes; i++) {
    int index = std::lower_bound(RowIn.begin(), RowIn.end(), des[i]) - RowIn.begin();
    U(index, i) = 1.0;
    RowOutWrong[index] = cre[i];
  }
  //igl::slice(inverseIn, VectorXi::LinSpaced(RowIn.size(), 0, RowIn.size() + 1), RowDes, inverseInU);
  MatrixXd inverseInU = inverseIn*U;
  MatrixXd Id = MatrixXd::Identity(ncre, ncre);
  MatrixXd detFactor = Id + newRow*inverseInU;
  MatrixXd detFactorInv, inverseOutWrong;

  Eigen::FullPivLU<MatrixXd> lub(detFactor);
  if (lub.isInvertible() ) {
    detFactorInv = lub.inverse();
    inverseOutWrong = inverseIn - ((inverseInU) * detFactorInv) * (newRow * inverseIn); 
    detValueOut = detValueIn * detFactor.determinant();
  }
  else {
    MatrixXd originalOrbs;
    Eigen::Map<VectorXi> Row(&RowIn[0], RowIn.size());
    igl::slice(Hforbs, Row, ColVec, originalOrbs);
    MatrixXd newOrbs = originalOrbs + U*newRow; 
    inverseOutWrong = newOrbs.inverse();
    detValueOut = newOrbs.determinant();
  }

  //now we need to reorder the inverse to correct the order of rows
  std::vector<int> order(RowOutWrong.size()), rcopy = RowOutWrong;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&rcopy](size_t i1, size_t i2) { return rcopy[i1] < rcopy[i2]; });
  Eigen::Map<VectorXi> orderVec(&order[0], order.size());
  igl::slice(inverseOutWrong, VectorXi::LinSpaced(ColVec.rows(), 0, ColVec.rows()), orderVec, inverseOut);
  
}



void CPSSlaterWalker::initUsingWave(CPSSlater& w, bool check) {

  AlphaTable.resize(w.determinants.size());
  BetaTable.resize(w.determinants.size());
  alphaDet.resize(w.determinants.size());
  betaDet.resize(w.determinants.size());
  
  AlphaOpen.clear(); AlphaClosed.clear(); BetaOpen.clear(); BetaClosed.clear();
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen , BetaClosed );

  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());
  Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
  Eigen::Map<VectorXi> RowBetaOpen (&BetaOpen[0] , BetaOpen.size());
  int norbs  = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta  = BetaClosed.size();

  vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

  vector<int> alphaRef0, betaRef0;
  w.determinants[0].getAlphaBeta(alphaRef0, betaRef0);

  for (int i=0; i<w.determinants.size(); i++) 
  {
    MatrixXd alpha, beta;
    MatrixXd alphainvCurrent, betainvCurrent;

    //Generate the alpha and beta strings for the wavefunction determinant
    vector<int> alphaRef, betaRef;
    w.determinants[i].getAlphaBeta(alphaRef, betaRef);
    Eigen::Map<VectorXi> ColAlpha(&alphaRef[0], alphaRef.size());
    Eigen::Map<VectorXi> ColBeta (&betaRef[0],  betaRef.size());


    if (i == 0)   
    //if (true)
    {
      igl::slice(w.HforbsA, RowAlpha, ColAlpha, alpha); //alpha = Hforbs(Row, Col)
      Eigen::FullPivLU<MatrixXd> lua(alpha);
      if (lua.isInvertible() || !check)
      {
        alphainv = lua.inverse();
        alphainvCurrent = alphainv;
        alphaDet[i] = lua.determinant();
      }
      else
      {
        cout << RowAlpha<<endl<<endl;
        cout << ColAlpha<<endl<<endl;
        cout << alpha<<endl<<endl;
        cout << lua.determinant()<<endl;
        cout << lua.inverse()<<endl;
        cout << "overlap with alpha determinant " << d << " not invertible" << endl;
        exit(0);
      }

      igl::slice(w.HforbsB, RowBeta, ColBeta, beta); //beta = Hforbs(Row, Col)
      Eigen::FullPivLU<MatrixXd> lub(beta);
      if (lub.isInvertible() || !check)
      {
        betainv = lub.inverse();
        betainvCurrent = betainv;
        betaDet[i] = lub.determinant();
      }
      else
      {
        cout << "overlap with beta determinant " << d << " not invertible" << endl;
        exit(0);
      }
    }
    else 
    {
      getOrbDiff(w.determinants[i], w.determinants[0], creA, desA, creB, desB);
      double alphaParity = w.determinants[0].parityA(creA, desA);
      double betaParity  = w.determinants[0].parityB(creB, desB);
      calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[i], creA, desA, RowAlpha, alphaRef0, w.HforbsA);
      calculateInverseDeterminantWithColumnChange(betainv , betaDet[0] , betainvCurrent , betaDet[i] , creB, desB, RowBeta, betaRef0, w.HforbsB);
      alphaDet[i] *= alphaParity;
      betaDet[i] *= betaParity;

    }
      //cout << alphainvCurrent<<endl<<endl;
      //cout << betainvCurrent<<endl<<endl;
      

    AlphaTable[i] = MatrixXd::Zero(AlphaOpen.size(), AlphaClosed.size()); //k-N x N  
    MatrixXd HfopenAlpha;
    igl::slice(w.HforbsA, RowAlphaOpen, ColAlpha, HfopenAlpha);
    AlphaTable[i] = HfopenAlpha*alphainvCurrent;

    BetaTable[i] = MatrixXd::Zero(BetaOpen.size(), BetaClosed.size()); //k-N x N  
    MatrixXd HfopenBeta;
    igl::slice(w.HforbsB, RowBetaOpen, ColBeta, HfopenBeta);
    BetaTable[i] = HfopenBeta*betainvCurrent;


  }
  //exit(0);
}

void   CPSSlaterWalker::exciteWalker(CPSSlater& w, int excite1, int excite2, int norbs)
{
  int I1 = excite1/(2*norbs), A1= excite1%(2*norbs);


  if (I1%2 == 0) updateA(I1/2, A1/2, w);
  else           updateB(I1/2, A1/2, w);

  if (excite2 != 0) {
    int I2 = excite2/(2*norbs), A2= excite2%(2*norbs);
    if (I2%2 == 0) updateA(I2/2, A2/2, w);
    else           updateB(I2/2, A2/2, w);
  }


}

double CPSSlaterWalker::getDetFactorA(int i, int a, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

  double p = 1.;
  if (doparity)
    d.parityA(a, i, p);

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = AlphaTable[i](tableIndexa, tableIndexi);
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }

  return p * detFactorNum / detFactorDen;
}

double CPSSlaterWalker::getDetFactorB(int i, int a, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();

  double p = 1.;
  if (doparity)
    d.parityB(a, i, p);

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = BetaTable[i](tableIndexa, tableIndexi);
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }

  return p * detFactorNum / detFactorDen;
}

double CPSSlaterWalker::getDetFactorA(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
  int tableIndexj = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), j) - AlphaClosed.begin();
  int tableIndexb = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), b) - AlphaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = (AlphaTable[i](tableIndexa, tableIndexi) * AlphaTable[i](tableIndexb, tableIndexj) - AlphaTable[i](tableIndexb, tableIndexi) * AlphaTable[i](tableIndexa, tableIndexj));
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return detFactorNum / detFactorDen;
}

double CPSSlaterWalker::getDetFactorB(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), a) - BetaOpen.begin();
  int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
  int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int i = 0; i < w.determinants.size(); i++)
  {
    double factor = (BetaTable[i](tableIndexa, tableIndexi) * BetaTable[i](tableIndexb, tableIndexj) - BetaTable[i](tableIndexb, tableIndexi) * BetaTable[i](tableIndexa, tableIndexj));
    detFactorNum += w.ciExpansion[i] * factor * alphaDet[i] * betaDet[i];
    detFactorDen += w.ciExpansion[i] * alphaDet[i] * betaDet[i];
  }
  return detFactorNum / detFactorDen;
}

double CPSSlaterWalker::getDetFactorAB(int i, int j, int a, int b, CPSSlater &w, bool doparity)
{
  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();
  int tableIndexj = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), j) - BetaClosed.begin();
  int tableIndexb = std::lower_bound(BetaOpen.begin(), BetaOpen.end(), b) - BetaOpen.begin();

  double detFactorNum = 0.0;
  double detFactorDen = 0.0;
  for (int I = 0; I < w.determinants.size(); I++)
  {
    double factor = AlphaTable[I](tableIndexa, tableIndexi) * BetaTable[I](tableIndexb, tableIndexj);
    detFactorNum += w.ciExpansion[I] * factor * alphaDet[I] * betaDet[I];
    detFactorDen += w.ciExpansion[I] * alphaDet[I] * betaDet[I];
  }
  return detFactorNum / detFactorDen;
}

double CPSSlaterWalker::getDetFactorA(vector<int>& iArray, vector<int>& aArray, CPSSlater &w, bool doparity)
{
  MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
  for (int i=0; i<iArray.size(); i++)
  {
    int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), iArray[i]) - AlphaClosed.begin();
    for (int a=0; a<aArray.size(); a++)
    {
      int tableIndexa = std::lower_bound(AlphaOpen  .begin(), AlphaOpen  .end(), aArray[a]) - AlphaOpen.begin();
      localDet(i,a) = AlphaTable[0](tableIndexi, tableIndexa);
    }
  }

  double p = 1.;
  Determinant dcopy = d;
  for (int i=0; i<iArray.size(); i++) 
  {
    if (doparity)
      dcopy.parityA(aArray[i], iArray[i], p);

    dcopy.setoccA(iArray[i], false);
    dcopy.setoccA(aArray[i], true);
  }

  double cpsFactor = 1.0;

  return p * cpsFactor * localDet.determinant() ;
}

double CPSSlaterWalker::getDetFactorB(vector<int>& iArray, vector<int>& aArray, CPSSlater &w, bool doparity)
{
  MatrixXd localDet = MatrixXd::Zero(aArray.size(), iArray.size());
  for (int i=0; i<iArray.size(); i++)
  {
    int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), iArray[i]) - BetaClosed.begin();
    for (int a=0; a<aArray.size(); a++)
    {
      int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), aArray[a]) - BetaOpen.begin();
      localDet(i,a) = BetaTable[0](tableIndexi, tableIndexa);
    }
  }

  double p = 1.;
  Determinant dcopy = d;
  for (int i=0; i<iArray.size(); i++) 
  {
    if (doparity)
      dcopy.parityB(aArray[i], iArray[i], p);

    dcopy.setoccB(iArray[i], false);
    dcopy.setoccB(aArray[i], true);
  }

  double cpsFactor = 1.0;

  return p * cpsFactor * localDet.determinant() ;
}



void CPSSlaterWalker::updateA(int i, int a, CPSSlater& w) {


  double p = 1.0;
  d.parityA(a, i, p);

  int tableIndexi = std::lower_bound(AlphaClosed.begin(), AlphaClosed.end(), i) - AlphaClosed.begin();
  int tableIndexa = std::lower_bound(AlphaOpen.begin(), AlphaOpen.end(), a) - AlphaOpen.begin();

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();
  vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);
  
  vector<int> alphaRef0, betaRef0;
  w.determinants[0].getAlphaBeta(alphaRef0, betaRef0);
  vector<int> cre(1, a), des(1, i);
  MatrixXd alphainvOld = alphainv;
  double alphaDetOld = alphaDet[0];
  Eigen::Map<Eigen::VectorXi> ColVec(&alphaRef0[0], alphaRef0.size());
  calculateInverseDeterminantWithRowChange(alphainvOld, alphaDetOld, alphainv,
                                           alphaDet[0], cre, des, ColVec, AlphaClosed, w.HforbsA);
  alphaDet[0] *= p;

  d.setoccA(i, false);
  d.setoccA(a, true);
  AlphaOpen.clear(); AlphaClosed.clear(); BetaOpen.clear(); BetaClosed.clear();
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());

  Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
  MatrixXd HfopenAlpha;
  igl::slice(w.HforbsA, RowAlphaOpen, ColVec, HfopenAlpha);
  AlphaTable[0] = HfopenAlpha * alphainv;
  
  for (int x=1; x<w.determinants.size(); x++) 
  {
    MatrixXd alphainvCurrent, betainvCurrent;

    vector<int> alphaRef, betaRef;
    w.determinants[x].getAlphaBeta(alphaRef, betaRef);
  
    getOrbDiff(w.determinants[x], w.determinants[0], creA, desA, creB, desB);
    double alphaParity = w.determinants[0].parityA(creA, desA);
    calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[x], creA, desA, RowAlpha, alphaRef0, w.HforbsA);
    alphaDet[x] *= alphaParity;

    MatrixXd HfopenAlpha;
    Eigen::Map<VectorXi> ColAlpha (&alphaRef[0],  alphaRef.size());
    igl::slice(w.HforbsA, RowAlphaOpen, ColAlpha, HfopenAlpha);
    AlphaTable[x] = HfopenAlpha * alphainvCurrent;
  }
  

}

void CPSSlaterWalker::updateA(int i, int j, int a, int b, CPSSlater& w) {

  double p = 1.0;
  Determinant dcopy = d;
  dcopy.parityA(a, i, p);
  dcopy.setoccA(i, false); dcopy.setoccA(a, true);
  dcopy.parityA(b, j, p);
  dcopy.setoccA(j, false); dcopy.setoccA(b, true);

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();
  vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);
  
  vector<int> alphaRef0, betaRef0;
  w.determinants[0].getAlphaBeta(alphaRef0, betaRef0);
  vector<int> cre(2, a), des(2, i); cre[1] = b; des[1] = j;
  MatrixXd alphainvOld = alphainv;
  double alphaDetOld = alphaDet[0];
  Eigen::Map<Eigen::VectorXi> ColVec(&alphaRef0[0], alphaRef0.size());
  calculateInverseDeterminantWithRowChange(alphainvOld, alphaDetOld, alphainv,
                                           alphaDet[0], cre, des, ColVec, AlphaClosed, w.HforbsA);
  alphaDet[0] *= p;

  d = dcopy;
  AlphaOpen.clear(); AlphaClosed.clear(); BetaOpen.clear(); BetaClosed.clear();
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());

  Eigen::Map<VectorXi> RowAlphaOpen(&AlphaOpen[0], AlphaOpen.size());
  MatrixXd HfopenAlpha;
  igl::slice(w.HforbsA, RowAlphaOpen, ColVec, HfopenAlpha);
  AlphaTable[0] = HfopenAlpha * alphainv;
  
  for (int x=1; x<w.determinants.size(); x++) 
  {
    MatrixXd alphainvCurrent, betainvCurrent;

    vector<int> alphaRef, betaRef;
    w.determinants[x].getAlphaBeta(alphaRef, betaRef);
  
    getOrbDiff(w.determinants[x], w.determinants[0], creA, desA, creB, desB);
    double alphaParity = w.determinants[0].parityA(creA, desA);
    calculateInverseDeterminantWithColumnChange(alphainv, alphaDet[0], alphainvCurrent, alphaDet[x], creA, desA, RowAlpha, alphaRef0, w.HforbsA);
    alphaDet[x] *= alphaParity;

    MatrixXd HfopenAlpha;
    Eigen::Map<VectorXi> ColAlpha (&alphaRef[0],  alphaRef.size());
    igl::slice(w.HforbsA, RowAlphaOpen, ColAlpha, HfopenAlpha);
    AlphaTable[x] = HfopenAlpha * alphainvCurrent;
  }
  

}

void CPSSlaterWalker::updateB(int i, int a, CPSSlater& w) {

  double p = 1.0;
  d.parityB(a, i, p);

  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();
  vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

  vector<int> alphaRef0, betaRef0;
  w.determinants[0].getAlphaBeta(alphaRef0, betaRef0);
  vector<int> cre(1, a), des(1, i);
  MatrixXd betainvOld = betainv;
  double betaDetOld = betaDet[0];
  Eigen::Map<Eigen::VectorXi> ColVec(&betaRef0[0], betaRef0.size());
  calculateInverseDeterminantWithRowChange(betainvOld, betaDetOld, betainv,
                                           betaDet[0], cre, des, ColVec, BetaClosed, w.HforbsB);
  betaDet[0] *= p;

  d.setoccB(i, false);
  d.setoccB(a, true);
  AlphaOpen.clear(); AlphaClosed.clear(); BetaOpen.clear(); BetaClosed.clear();
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());

  Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
  MatrixXd HfopenBeta;
  igl::slice(w.HforbsB, RowBetaOpen, ColVec, HfopenBeta);
  BetaTable[0] = HfopenBeta * betainv;

  for (int x=1; x<w.determinants.size(); x++) 
  {
    MatrixXd betainvCurrent;

    vector<int> alphaRef, betaRef;
    w.determinants[x].getAlphaBeta(alphaRef, betaRef);
  
    getOrbDiff(w.determinants[x], w.determinants[0], creA, desA, creB, desB);
    double betaParity = w.determinants[0].parityB(creB, desB);
    calculateInverseDeterminantWithColumnChange(betainv, betaDet[0], betainvCurrent, betaDet[x], creB, desB, RowBeta, betaRef0, w.HforbsB);
    betaDet[x] *= betaParity;

    MatrixXd HfopenBeta;
    Eigen::Map<VectorXi> ColBeta (&betaRef[0],  betaRef.size());
    igl::slice(w.HforbsB, RowBetaOpen, ColBeta, HfopenBeta);
    BetaTable[x] = HfopenBeta*betainvCurrent;

  }
  
}

void CPSSlaterWalker::updateB(int i, int j, int a, int b, CPSSlater& w) {

  double p = 1.0;
  Determinant dcopy = d;
  dcopy.parityB(a, i, p);
  dcopy.setoccB(i, false); dcopy.setoccB(a, true);
  dcopy.parityB(b, j, p);
  dcopy.setoccB(j, false); dcopy.setoccB(b, true);

  int tableIndexi = std::lower_bound(BetaClosed.begin(), BetaClosed.end(), i) - BetaClosed.begin();
  int tableIndexa = std::lower_bound(BetaOpen  .begin(), BetaOpen  .end(), a) - BetaOpen.begin();

  int norbs = Determinant::norbs;
  int nalpha = AlphaClosed.size();
  int nbeta = BetaClosed.size();
  vector<int> creA(nalpha, -1), desA(nalpha, -1), creB(nbeta, -1), desB(nbeta, -1);

  vector<int> alphaRef0, betaRef0;
  w.determinants[0].getAlphaBeta(alphaRef0, betaRef0);
  vector<int> cre(2, a), des(2, i); cre[1] = b; des[1] = j;
  MatrixXd betainvOld = betainv;
  double betaDetOld = betaDet[0];
  Eigen::Map<Eigen::VectorXi> ColVec(&betaRef0[0], betaRef0.size());
  calculateInverseDeterminantWithRowChange(betainvOld, betaDetOld, betainv,
                                           betaDet[0], cre, des, ColVec, BetaClosed, w.HforbsB);
  betaDet[0] *= p;

  d = dcopy;
  AlphaOpen.clear(); AlphaClosed.clear(); BetaOpen.clear(); BetaClosed.clear();
  d.getOpenClosedAlphaBeta(AlphaOpen, AlphaClosed, BetaOpen, BetaClosed);
  Eigen::Map<VectorXi> RowAlpha(&AlphaClosed[0], AlphaClosed.size());
  Eigen::Map<VectorXi> RowBeta (&BetaClosed[0] , BetaClosed.size());

  Eigen::Map<VectorXi> RowBetaOpen(&BetaOpen[0], BetaOpen.size());
  MatrixXd HfopenBeta;
  igl::slice(w.HforbsB, RowBetaOpen, ColVec, HfopenBeta);
  BetaTable[0] = HfopenBeta * betainv;

  for (int x=1; x<w.determinants.size(); x++) 
  {
    MatrixXd betainvCurrent;

    vector<int> alphaRef, betaRef;
    w.determinants[x].getAlphaBeta(alphaRef, betaRef);
  
    getOrbDiff(w.determinants[x], w.determinants[0], creA, desA, creB, desB);
    double betaParity = w.determinants[0].parityB(creB, desB);
    calculateInverseDeterminantWithColumnChange(betainv, betaDet[0], betainvCurrent, betaDet[x], creB, desB, RowBeta, betaRef0, w.HforbsB);
    betaDet[x] *= betaParity;

    MatrixXd HfopenBeta;
    Eigen::Map<VectorXi> ColBeta (&betaRef[0],  betaRef.size());
    igl::slice(w.HforbsB, RowBetaOpen, ColBeta, HfopenBeta);
    BetaTable[x] = HfopenBeta*betainvCurrent;

  }
  
}