#include "Residuals.h"
#include "Determinants.h"
#include "global.h"
#include "integral.h"
#include "input.h"

void applyProjector(
    MatrixXcd& bra,
    vector<MatrixXcd>& ketvec,
    vector<complex<double> >& coeffs,
    int Sz,
    int ngrid) {

  int norbs = Determinant::norbs;
  double m = 1.*Sz;

  complex<double> iImag(0, 1.0);
  coeffs.resize(ngrid, 0.0);
  ketvec.resize(ngrid, bra);

  for (int g = 0; g<ngrid; g++) {
    DiagonalMatrix<complex<double>, Dynamic> phi(2*norbs);
    double angle = g*2*M_PI/ngrid;
    for (int i=0; i<norbs; i++) {
      phi.diagonal()[i]       = exp(iImag*angle/2.);
      phi.diagonal()[i+norbs] = exp(-iImag*angle/2.);
    }
    ketvec[g] = phi * bra;
    coeffs[g] = exp(-iImag*angle*m)/ngrid;

    //ketvec[2*g+1] = (phi * bra).conjugate();
    //coeffs[2*g+1] = exp(iImag*angle*m)/ngrid/2.0;
  }
}


void fillJastrowfromWfn(MatrixXd& Jtmp, VectorXd& JA) {
  int norbs = Determinant::norbs;
  
  for (int i=0; i<2*norbs; i++) {
    int I = (i/2) + (i%2)*norbs;
    for (int j=0; j<i; j++) {
      int J = (j/2) + (j%2)*norbs;
      
      JA(index(J, I)) = log(Jtmp(i, j))/2.0;
    }
    JA(index(I, I)) = log(Jtmp(i,i));///2;
  }
  
}

void Residuals::updateOrbitals(VectorXd& braReal) {

  for (int i=0; i<2*norbs; i++) 
    for (int j=0; j<bra.cols(); j++) 
      bra(i,j) = complex<double>(braReal(2*(i*bra.cols()+j)), braReal(2*(i*bra.cols()+j)+1));

  applyProjector(bra, ket, coeffs, Sz, ngrid);      
}

int Residuals::getOrbitalResidue(const VectorXd& braReal, VectorXd& residueReal) {
  int norbs = Determinant::norbs;

  //copy the braReal to the Bra;
  for (int i=0; i<norbs; i++) 
    for (int j=0; j<bra.cols(); j++) 
      bra(i,j) = complex<double>(braReal(2*i*bra.cols()+j), braReal(2*i*bra.cols()+j+1));

  MatrixXcd residue = 0*bra; //zero out the residue

  MatrixXcd S;
  double detovlp;
  for (int g = 0; g<ket.size(); g++) {
    S = bra.adjoint()*ket[g];
    detovlp += (S.determinant() * coeffs[g]).real();
  }


  for (int g = 0; g<ket.size(); g++) 
    getOrbResidueSingleKet(detovlp, bra, ket[g], coeffs[g], residue);

  return 0;

}

void Residuals::getOrbResidueSingleKet(
    double detovlp,
    MatrixXcd& bra,
    MatrixXcd& ket,
    complex<double> coeff,
    MatrixXcd& residue) {


  MatrixXcd LambdaD, LambdaC, S;

  DiagonalXd diagcre(2*norbs),
      diagdes(2*norbs);

  complex<double> Energy;
  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
      double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
      if (abs(integral) < schd.epsilon) continue;
        
      double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, Jastrow);
      LambdaD = diagdes*ket;
      LambdaC = diagcre*bra;
        
      S = LambdaC.adjoint()*LambdaD;

      factor *= integral /detovlp;

      std::complex<double> Sdet = S.determinant();
      MatrixXcd Sinv = S.inverse();
      
      complex<double> rdm = ((LambdaD.row(orb2) * Sinv)*LambdaC.adjoint().col(orb1))(0,0);

      //d[det(S) LD Sinv LC]/d[theta]
      //Term1 det(S) LD Sinv d[LC]/d[theta]
      //residue.row(orb1) += factor*Sdet * (LambdaD.row(orb2) * Sinv) * diagcre(orb1);

      //Term2 det(S) Tr(Sinv * d[S]/d[theta]) * (LD Sinv LC)      
      //residue += (factor * Sdet * rdm) * (Sinv.transpose() * LambdaD);

      //term3 det(S) LD Sinv d[S]/d[theta] Sinv LC
      //residue += factor * Sdet * 
      //**don't need to calculate the entire RDM, should make it more efficient
      Energy += rdm * integral * factor * Sdet / detovlp;
      //Energy += rdm * integral * S.determinant() / detovlp;
    }
    

  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
            
          int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
          int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
          int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
          int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
          double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
          if (abs(integral) < schd.epsilon) continue;
            
          double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, Jastrow);
          LambdaD = diagdes*ket;
          LambdaC = diagcre*bra;
          S = LambdaC.adjoint()*LambdaD;

          //**don't need to calculate the entire RDM, should make it more efficient
          complex<double> rdm1 = ((LambdaD.row(orb4) * S.inverse()) * LambdaC.adjoint().col(orb1));
          complex<double> rdm2 = ((LambdaD.row(orb3) * S.inverse()) * LambdaC.adjoint().col(orb2));
          complex<double> rdm3 = ((LambdaD.row(orb3) * S.inverse()) * LambdaC.adjoint().col(orb1));
          complex<double> rdm4 = ((LambdaD.row(orb4) * S.inverse()) * LambdaC.adjoint().col(orb2));
            
          Energy += (rdm1*rdm2 - rdm3*rdm4) * integral * factor * S.determinant() / detovlp;
        }
    }

}


int Residuals::getJastrowResidue(const VectorXd& JA, VectorXd& residue)
{
  int norbs = Determinant::norbs;
  
  //calculate <bra|P|ket> and
  double detovlp = 0.0;
  VectorXd NiNjRDM(2*norbs*(2*norbs+1)/2); NiNjRDM.setZero();
  MatrixXcd mfRDM(2*norbs, 2*norbs);
  
  for (int g = 0; g<ket.size(); g++) {
    MatrixXcd S = bra.adjoint()*ket[g];
    complex<double> Sdet = S.determinant();
    detovlp += (S.determinant() * coeffs[g]).real();

    mfRDM = ((ket[g] * S.inverse())*bra.adjoint()); 
    
    for (int i=0; i<2*norbs; i++) {
      for (int j=0; j<=i; j++) {
        int index = i*(i+1)/2+j;
        if (i == j)
          NiNjRDM(index) += (mfRDM(j, i)* conj(coeffs[g]) * Sdet).real();
        else
          NiNjRDM(index) += ((mfRDM(j,j)*mfRDM(i,i) - mfRDM(i,j) * mfRDM(j,i))* conj(coeffs[g]) * Sdet).real();
      }
    }
  }
  NiNjRDM /= detovlp;
  
  double Energy = 0.;
  complex<double> cEnergy = 0.0;
  VectorXd intermediateResidue(2*norbs*(2*norbs+1)/2); intermediateResidue.setZero();

  for (int g = 0; g<ket.size(); g++)  {
    cEnergy += (getResidueSingleKet(JA, intermediateResidue,
                                    detovlp, coeffs[g], const_cast<MatrixXcd&>(bra),
                                    ket[g]) * conj(coeffs[g]));
  }
  Energy = cEnergy.real();
  
  residue = intermediateResidue - Energy *NiNjRDM;

  const_cast<double&>(this->E0) = Energy;
  return 0;
}
  
complex<double> Residuals::getResidueSingleKet (const VectorXd& JA,
                                                VectorXd& residue, double detovlp, complex<double> coeff,
                                                MatrixXcd& bra, const MatrixXcd& ket) const
{
  int nelec = nalpha+nbeta;
  Matrix4cd rdmResidue4;
  Matrix3cd rdmResidue3;
  Matrix2cd rdmResidue2;
  vector<int> Rows(4), Cols(4);

  MatrixXcd LambdaD = bra, LambdaC = ket; //just initializing  
  MatrixXcd S(bra.cols(), ket.cols());

  DiagonalXd diagcre(2*norbs),
      diagdes(2*norbs);
    
  complex<double> Energy = 0.0;
  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
      double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
      if (abs(integral) < schd.epsilon) continue;
        
      complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
      LambdaD = diagdes*ket;
      LambdaC = diagcre*bra;
      S = LambdaC.adjoint()*LambdaD;

      /*
      Eigen::FullPivLU<MatrixXcd> lu(S);
      MatrixXcd rdm = LambdaD * lu.solve(LambdaC.adjoint());
      //MatrixXcd Sinv = S.inverse();
      complex<double> Sdet = 1.0;
      for (int i=0; i<nelec; i++)
        Sdet *= lu.matrixLU()(i,i);
      */
      complex<double> Sdet = S.determinant();
      MatrixXcd rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
      
      factor *= Sdet/detovlp;

      //MatrixXcd rdm = (LambdaD * Sinv)*LambdaC.adjoint();

      Energy += rdm(orb2, orb1) * integral * factor;
      vector<double> graddiag(2*norbs, 0.0), graddiag2(2*norbs, 0.0); double rdmscale = 1.0;

      complex<double> res;
      for (int orbn = 0; orbn < 2*norbs; orbn++) {
        for (int orbm = 0; orbm < orbn; orbm++) {
          res = getRDMExpectation(rdm, orbn, orbm, orb1, orb2, rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
          residue(orbn*(orbn+1)/2 + orbm) += (res * factor * integral * conj(coeff)).real();
        }
        res = getRDMExpectation(rdm, orbn, orb1, orb2, rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
        residue(orbn*(orbn+1)/2 + orbn) += (res * factor * integral * conj(coeff)).real();
      }

    }
    
    
  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
            
          int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
          int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
          int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
          int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
          double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
          if (abs(integral) < schd.epsilon) continue;
            
          complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
          LambdaD = diagdes*ket;
          LambdaC = diagcre*bra;
          S = LambdaC.adjoint()*LambdaD;

          /*
          Eigen::FullPivLU<MatrixXcd> lu(S);
          MatrixXcd rdm = LambdaD * lu.solve(LambdaC.adjoint());
          complex<double> Sdet = 1.0;
          for (int i=0; i<nelec; i++)
            Sdet *= lu.matrixLU()(i,i);
          */
          complex<double> Sdet = S.determinant();
          MatrixXcd rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
          
          complex<double> rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
          factor *= Sdet/detovlp;
          Energy += rdmval * integral * factor;

          complex<double> res;
          for (int orbn = 0; orbn < 2*norbs; orbn++) {
            for (int orbm = 0; orbm < orbn; orbm++) {
              res = getRDMExpectation(rdm, orbn, orbm, orb1, orb2, orb3, orb4, rdmResidue4, rdmResidue3,
                                      rdmResidue2, Rows, Cols);
              residue(orbn*(orbn+1)/2 + orbm) += (res*factor*integral * conj(coeff)).real();
            }
            res = getRDMExpectation(rdm, orbn, orb1, orb2, orb3, orb4, rdmResidue4, rdmResidue3,
                                    rdmResidue2, Rows, Cols);
            residue(orbn*(orbn+1)/2 + orbn) += (res*factor*integral * conj(coeff)).real();
          }
        }
    }

  return Energy;
}


//obviously the energy should be real
double Residuals::Energy() const
{
  //calculate <bra|P|ket> and

  MatrixXcd S;
  double detovlp;
  for (int g = 0; g<ket.size(); g++) {
    S = bra.adjoint()*ket[g];
    detovlp += (S.determinant() * coeffs[g]).real();
  }

  double Energy = 0.0;
  for (int g = 0; g<ket.size(); g++) {
    Energy += (energyContribution(detovlp, bra, ket[g]) * coeffs[g]).real();
    //cout <<"bra "<<endl<< bra<<endl;
    //cout <<"ket "<<endl<<ket[g]<<endl;
    //cout << "coeff "<<coeffs[g]<<endl;
    //cout << "in energy "<<g<<"  "<<Energy<<endl;
  }
  return Energy;
}

complex<double> Residuals::energyContribution(
    double &detovlp,
    const MatrixXcd& bra,
    const MatrixXcd& ket) const
{

  MatrixXcd LambdaD;
  Matrix<complex<double>, Dynamic, Dynamic> LambdaC, S;

  DiagonalXd diagcre(2*norbs),
      diagdes(2*norbs);
    
  complex<double> Energy = 0.0;
  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
      double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
      if (abs(integral) < schd.epsilon) continue;
        
      double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, Jastrow);
      LambdaD = diagdes*ket;

      LambdaC = diagcre*bra;
        
      S = LambdaC.adjoint()*LambdaD;

      //factor *= S.determinant()/detovlp;
      //**don't need to calculate the entire RDM, should make it more efficient
      complex<double> rdm = ((LambdaD.row(orb2) * S.inverse())*LambdaC.adjoint().col(orb1))(0,0);
      Energy += rdm * integral * factor * S.determinant() / detovlp;
      //Energy += rdm * integral * S.determinant() / detovlp;
    }
    

  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
            
          int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
          int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
          int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
          int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
          double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
          if (abs(integral) < schd.epsilon) continue;
            
          double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, Jastrow);
          LambdaD = diagdes*ket;
          LambdaC = diagcre*bra;
          S = LambdaC.adjoint()*LambdaD;

          complex<double> rdm1 = ((LambdaD.row(orb4) * S.inverse()) * LambdaC.adjoint().col(orb1));
          complex<double> rdm2 = ((LambdaD.row(orb3) * S.inverse()) * LambdaC.adjoint().col(orb2));
          complex<double> rdm3 = ((LambdaD.row(orb3) * S.inverse()) * LambdaC.adjoint().col(orb1));
          complex<double> rdm4 = ((LambdaD.row(orb4) * S.inverse()) * LambdaC.adjoint().col(orb2));
            
          Energy += (rdm1*rdm2 - rdm3*rdm4) * integral * factor * S.determinant() / detovlp;
        }
    }

  return Energy;
}


int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}


//term is orb1^dag orb2
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int norbs,
                           const VectorXd&JA) {
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  double factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;
}

//term is orb1^dag orb2^dag orb3 orb4
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int orb3,
                           int orb4,
                           int norbs,
                           const VectorXd&JA) {

  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  double factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                       -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
}



//N_orbn N_orbm orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orbm, int orb1, int orb2, int orb3, int orb4,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {

  complex<double> contribution = 0;
  bool calc4rdm = true;

  if (orbm == orb2) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2 && orbm == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1 && orbm == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc4rdm = false;
  }

  if (calc4rdm) {
    rows[0] = orbm; rows[1] = orbn; rows[2] = orb3; rows[3] = orb4;
    cols[0] = orbm; cols[1] = orbn; cols[2] = orb2; cols[3] = orb1;
    
    Slice(rdm, rows, cols, rdmval4);
    contribution =  rdmval4.determinant();
  }
  
  return contribution;
}


//N_orbn orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orb1, int orb2, int orb3, int orb4,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {


  complex<double> contribution = 0;
  bool calc3rdm = true;

  
  if (orbn == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb1; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc3rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb2; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc3rdm = false;
  }

  if (calc3rdm) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orbn; cols[1] = orb2; cols[2] = orb1;
    
    Slice(rdm, rows, cols, rdmval3);
    contribution =  rdmval3.determinant();
  }
  
  return contribution;
}

//N_orbn N_orbm orb1^dag orb2
complex<double> getRDMExpectation(const MatrixXcd& rdm, int orbn, int orbm, int orb1, int orb2,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {
  
  rows[0] = orbn; rows[1] = orbm; rows[2] = orb2;
  cols[0] = orbn; cols[1] = orbm; cols[2] = orb1;

  complex<double> contribution;
  Slice(rdm, rows, cols, rdmval3);
  contribution =  rdmval3.determinant();
  
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
  }
  return contribution;
}



//N_orbn orb1^dag orb2
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orb1, int orb2,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {

  rows[0] = orbn; rows[1] = orb2;
  cols[0] = orbn; cols[1] = orb1;

  complex<double> contribution ;
  Slice(rdm, rows, cols, rdmval2);
  contribution =  rdmval2.determinant();
  
  if (orbn == orb1) {
    contribution += rdm(orb2, orbn);
  }

  return contribution;
}
