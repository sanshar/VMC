#include "MoDeterminants.h"
#include "integral.h"
#include "Determinants.h"

using namespace Eigen;
using namespace std;

double MoDeterminant::Overlap(Determinant& d) {

  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  return Overlap(alpha, beta);
}

void MoDeterminant::HamAndOvlp(Determinant& d, 
			       double& ovlp, double& ham,
			       oneInt& I1, twoInt& I2, double& coreE) {
  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  HamAndOvlp(alpha, beta, ovlp, ham, I1, I2, coreE);

}


double MoDeterminant::Overlap(vector<int>& alpha, vector<int>& beta) {
  MatrixXd DetAlpha = MatrixXd::Zero(nalpha, nalpha);
  MatrixXd DetBeta  = MatrixXd::Zero(nbeta,  nbeta);
  
  //<psi1 | psi2> = det(phi1^dag phi2)
  //in out case psi1 is a simple occupation number determinant
  for (int i=0; i<alpha.size(); i++)
    for (int j=0; j<DetAlpha.cols(); j++) 
      DetAlpha(i,j) = AlphaOrbitals(alpha[i], j);

  
  for (int i=0; i<beta.size(); i++)
    for (int j=0; j<DetBeta.cols(); j++)
      DetBeta(i,j) = BetaOrbitals(beta[i], j);

  double parity = 1.0;
  for (int i=0; i<beta.size(); i++)
    for (int j=0; j<alpha.size(); j++) 
      if (alpha[j] > beta[i]) parity *= -1.0;
  return parity*DetAlpha.determinant()*DetBeta.determinant();
}

void getunoccupiedOrbs(vector<int>& alpha, vector<int>& alphaOpen, int& norbs) {
  int i=0, j=0, index=0;
  while (i<alpha.size() && j < norbs) {
    if (alpha[i] < j) i++;
    else if (j<alpha[i]) {
      alphaOpen[index] = j;
      index++; j++;
    }
    else {i++;j++;}
  }
  while(j<norbs) {
    alphaOpen[index] = j;
    index++; j++;
  }
}

void MoDeterminant::HamAndOvlp(vector<int>& alpha, vector<int>& beta, 
				 double& ovlp, double& ham,
				 oneInt& I1, twoInt& I2, double& coreE) {

  int norbs = MoDeterminant::norbs;
  vector<int> alphaOpen(norbs-alpha.size(),0), betaOpen(norbs-beta.size(),0);
  std::sort(alpha.begin(), alpha.end());
  std::sort(beta.begin() , beta.end() );

  getunoccupiedOrbs(alpha, alphaOpen, norbs);
  getunoccupiedOrbs(beta,  betaOpen,  norbs);


  //noexcitation
  {
    double E0 = coreE;
    for (int i=0; i<alpha.size(); i++) {
      int I = 2*alpha[i];
      E0 += I1(I, I);
      
      for (int j=i+1; j<alpha.size(); j++) {
	int J = 2*alpha[j];
	E0 += I2(I, I, J, J) - I2(I, J, I, J);
      }
      
      for (int j=0; j<beta.size(); j++) {
	int J = 2*beta[j]+1;
	E0 += I2(I, I, J, J);
      }
    }
    
    for (int i=0; i<beta.size(); i++) {
      int I = 2*beta[i]+1;
      E0 += I1(I, I);
      
      for (int j=i+1; j<beta.size(); j++) {
	int J = 2*beta[j]+1;
	E0 += I2(I, I, J, J) - I2(I, J, I, J);
      }
    }
    ovlp = Overlap(alpha, beta);
    ham  = ovlp*E0;
  }

  //Single excitation alpha
  {
    vector<int> alphaCopy = alpha;
    for (int i=0; i<alpha.size(); i++)
      for (int a=0; a<alphaOpen.size(); a++) {
	int I = 2*alpha[i], A = 2*alphaOpen[a];
	
	alphaCopy[i] = alphaOpen[a];
	double tia = I1(I, A);

	for (int j=0; j<alpha.size(); j++) {
	  int J = 2*alpha[j];
	  tia += I2(A, I, J, J) - I2(A, J, J, I);
	}
	for (int j=0; j<beta.size(); j++) {
	  int J = 2*beta[j]+1;
	  tia += I2(A, I, J, J);
	}

	if (abs(tia) > 1.e-8) 	
	  ham += tia*Overlap(alphaCopy, beta);
	alphaCopy[i] = alpha[i];
      }
  }

  //Single excitation beta
  {
    vector<int> betaCopy = beta;
    for (int i=0; i<beta.size(); i++)
      for (int a=0; a<betaOpen.size(); a++) {
	int I = 2*beta[i]+1, A = 2*betaOpen[a]+1;
	
	betaCopy[i] = betaOpen[a];
	double tia = I1(I, A);

	for (int j=0; j<beta.size(); j++) {
	  int J = 2*beta[j]+1;
	  tia += I2(A, I, J, J) - I2(A, J, J, I);
	}
	for (int j=0; j<alpha.size(); j++) {
	  int J = 2*alpha[j];
	  tia += I2(A, I, J, J);
	}

	if (abs(tia) > 1.e-8) 	
	  ham += tia*Overlap(alpha, betaCopy);

	betaCopy[i] = beta[i];
      }
  }

  

  //alpha-alpha
  {
    vector<int> alphaCopy = alpha;
    for (int i=0; i<alpha.size(); i++)
    for (int a=0; a<alphaOpen.size(); a++) {
      int I = 2*alpha[i], A = 2*alphaOpen[a];
      alphaCopy[i] = alphaOpen[a];
      
      for (int j=i+1; j<alpha.size(); j++)
      for (int b=a+1; b<alphaOpen.size(); b++) {
	int J = 2*alpha[j], B = 2*alphaOpen[b];
	
	double tiajb = I2(A, I, B, J) - I2(A, J, B, I);
	alphaCopy[j] = alphaOpen[b];
	
	if (abs(tiajb) > 1.e-8) 	
	  ham += tiajb*Overlap(alphaCopy, beta);
	alphaCopy[j] = alpha[j];
      }
      alphaCopy[i] = alpha[i];
    }
  }

  //alpha-beta
  {
    vector<int> alphaCopy = alpha;
    vector<int> betaCopy = beta;
    for (int i=0; i<alpha.size(); i++)
    for (int a=0; a<alphaOpen.size(); a++) {
      int I = 2*alpha[i], A = 2*alphaOpen[a];
      alphaCopy[i] = alphaOpen[a];
      for (int j=0; j<beta.size(); j++)
      for (int b=0; b<betaOpen.size(); b++) {
	int J = 2*beta[j]+1, B = 2*betaOpen[b]+1;
	
	double tiajb = I2(A, I, B, J);
	betaCopy[j] = betaOpen[b];
	
	if (abs(tiajb) > 1.e-8) 	
	  ham += tiajb*Overlap(alphaCopy, betaCopy);
	betaCopy[j] = beta[j];
      }
      alphaCopy[i] = alpha[i];
    }
  }


  //beta-beta
  {
    vector<int> betaCopy = beta;
    for (int i=0; i<beta.size(); i++)
    for (int a=0; a<betaOpen.size(); a++) {
      int I = 2*beta[i]+1, A = 2*betaOpen[a]+1;
      betaCopy[i] = betaOpen[a];
      
      for (int j=i+1; j<beta.size(); j++)
      for (int b=a+1; b<betaOpen.size(); b++) {
	int J = 2*beta[j]+1, B = 2*betaOpen[b]+1;
	
	double tiajb = I2(A, I, B, J) - I2(A, J, B, I);
	betaCopy[j] = betaOpen[b];
	
	if (abs(tiajb) > 1.e-8) 	
	  ham += tiajb*Overlap(alpha, betaCopy);
	betaCopy[j] = beta[j];
      }

      betaCopy[i] = beta[i];
    }
  }

  
}
  

//<d|*this>
double MoDeterminant::Overlap(MoDeterminant& d) {

  MatrixXd DetAlpha = d.AlphaOrbitals.transpose()*AlphaOrbitals;
  MatrixXd DetBeta  = d.BetaOrbitals.transpose() *BetaOrbitals;

  return DetAlpha.determinant()*DetBeta.determinant();
 
}
