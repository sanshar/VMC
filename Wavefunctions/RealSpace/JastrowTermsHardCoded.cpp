#include "JastrowTermsHardCoded.h"
#include "JastrowBasis.h"
#include "input.h"
#include "global.h"
#include <vector>

inline double fastpow(double a, double b) {
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
  return u.d;
}

bool electronsOfCorrectSpin(const int& i, const int& j, const int& ss) {
  if ( ss == 2 ||  //applies to any term
       (i/rDeterminant::nalpha == j/rDeterminant::nalpha && ss == 1) ||
       (i/rDeterminant::nalpha != j/rDeterminant::nalpha && ss == 0) ) 
    return true;
  else
    return false;
}


void scaledRij(double& rij, double& rijbar,
               double& df, double& d2f) {
  rijbar = rij/(1+rij);
  df     = 1. /pow(1+rij,2);
  d2f    = -2./pow(1+rij,3);
}

/*
void scaledRij(double& rij, double& rijbar,
               double& df, double& d2f, int n) {
  double beta = .5;
  double poly = pow(rij, n);
  double exponent = exp(-beta*rij*rij);
  rijbar = poly*exponent;
  df     = (n*poly/rij - 2*beta*rij*poly)*exponent;
  d2f    = (n*(n-1)*poly/rij/rij
            - (2*beta*n  + 2*beta*(n+1))*poly
            + 4*beta*beta*rij*rij*poly)*exponent;
}
*/

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowEEValue(int i, int j, int maxQ,
                      const vector<Vector3d>& r,
                      const VectorXd& params,
                      int startIndex,
                      int ss) {
  double value = 0.0;
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double rijbar, df, d2f;
      scaledRij(rij, rijbar, df, d2f);

      for (int n = 1; n <= maxQ; n++) 
        value += pow(rijbar, n) * params[startIndex + n - 1];

    }
  }
  return value;
}


//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowEEValueGrad(int i, int j, int maxQ,
                          const vector<Vector3d>& r,
                          Vector3d& grad,
                          const VectorXd& params,
                          int startIndex,
                          int ss) {
  double value = 0.0;
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double rijbar, df, d2f;
      scaledRij(rij, rijbar, df, d2f);
    
      for (int n = 1; n <= maxQ; n++) {
        const double& factor = params[startIndex + n - 1];
        double val = pow(rijbar, n);

        value   += val * factor;      
        grad(0) += factor * (n * val / rijbar) * df * xij/rij;
        grad(1) += factor * (n * val / rijbar) * df * yij/rij;
        grad(2) += factor * (n * val / rijbar) * df * zij/rij;

      }
    }
  }
  return value;
}



//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
void JastrowEE(int i, int j, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex,
               int ss) { 
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double rijbar, df, d2f;
      scaledRij(rij, rijbar, df, d2f);
    
      for (int n = 1; n <= maxQ; n++) {
        double value = pow(rijbar, n);
      
        values[startIndex + n - 1] += factor*value;

        double gradx = (n * value / rijbar) * df * xij/rij;
        double grady = (n * value / rijbar) * df * yij/rij;
        double gradz = (n * value / rijbar) * df * zij/rij;

        double laplacian = (n * (n-1) * value / rijbar / rijbar) * df * df
            + (n * value / rijbar) * ( d2f + df * 2 / rij);

        gx    (i, startIndex + n - 1) += factor*gradx; 
        gy    (i, startIndex + n - 1) += factor*grady;
        gz    (i, startIndex + n - 1) += factor*gradz;      
        laplace(i, startIndex + n - 1) += factor*laplacian;

        gx    (j, startIndex + n - 1) -= factor*gradx; 
        gy    (j, startIndex + n - 1) -= factor*grady;
        gz    (j, startIndex + n - 1) -= factor*gradz;      
        laplace(j, startIndex + n - 1) += factor*laplacian;
      
      }
    }
  }
}

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
double JastrowENValue(int i, int maxQ,
                      const vector<Vector3d>& r,
                      const VectorXd& params,
                      int startIndex) {
  vector<double>& Ncharge = schd.Ncharge;
  double value = 0.0;
  int natom = Ncharge.size();
  
  for (int N=0; N<natom; N++)
  {
    const Vector3d& di = r[i], &dN = schd.Ncoords[N];
    
    double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
    xiN = di[0] - dN[0];
    yiN = di[1] - dN[1];
    ziN = di[2] - dN[2];
    riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

    double riNbar, df, d2f;
    scaledRij(riN, riNbar, df, d2f);

    for (int n = 1; n <= maxQ; n++) 
      value += pow(riNbar, n) * params[startIndex + (n - 1) + N * maxQ];
  }
  return value;
}


double JastrowENValueGrad(int i, int maxQ,
                          const vector<Vector3d>& r,
                          Vector3d& grad,
                          const VectorXd& params,
                          int startIndex) {
  double value = 0.0;
  vector<double>& Ncharge = schd.Ncharge;
  
  for (int N=0; N<Ncharge.size(); N++)
  {
    const Vector3d& di = r[i], &dN = schd.Ncoords[N];
    
    double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
    xiN = di[0] - dN[0];
    yiN = di[1] - dN[1];
    ziN = di[2] - dN[2];
    riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

    double riNbar, df, d2f;
    scaledRij(riN, riNbar, df, d2f);
    
    for (int n = 1; n <= maxQ; n++) {
      const double& factor = params[startIndex + n - 1 + N * maxQ];
      double val = pow(riNbar, n);
      
      value   += factor * val;
      grad(0) += factor * (n * val / riNbar) * df * xiN/riN;
      grad(1) += factor * (n * val / riNbar) * df * yiN/riN;
      grad(2) += factor * (n * val / riNbar) * df * ziN/riN;
    }
  }
  return value;
}


//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
void JastrowEN(int i, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex) {
  vector<double>& Ncharge = schd.Ncharge;
  
  for (int N=0; N<Ncharge.size(); N++)
  {
    const Vector3d& di = r[i], &dN = schd.Ncoords[N];
    
    double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
    xiN = di[0] - dN[0];
    yiN = di[1] - dN[1];
    ziN = di[2] - dN[2];
    riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

    double riNbar, df, d2f;
    scaledRij(riN, riNbar, df, d2f);
    
    for (int n = 1; n <= maxQ; n++) {
      double value = pow(riNbar, n);
      
      values[startIndex + n - 1 + N * maxQ] += factor*value;

      double gradx = (n * value / riNbar) * df * xiN/riN;
      double grady = (n * value / riNbar) * df * yiN/riN;
      double gradz = (n * value / riNbar) * df * ziN/riN;

      double laplacian = (n * (n-1) * value / riNbar / riNbar) * df * df
          + (n * value / riNbar) * ( d2f + df * 2 / riN);

      gx    (i, startIndex + n - 1 + N * maxQ) += factor*gradx; 
      gy    (i, startIndex + n - 1 + N * maxQ) += factor*grady;
      gz    (i, startIndex + n - 1 + N * maxQ) += factor*gradz;      
      laplace(i, startIndex + n - 1+ N * maxQ) += factor*laplacian;
    }
  }
}


double JastrowEENValueGrad(int i, int j, int maxQ,
                         const vector<Vector3d>& r,
                         Vector3d& grad,
                         const VectorXd& params,
                         int startIndex,
                         int ss) {
  vector<double>& Ncharge = schd.Ncharge;
  double value = 0.0;
  int index = 0;
  
  for (int N=0; N<Ncharge.size(); N++)
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j], &dN = schd.Ncoords[N];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
      xiN = di[0] - dN[0];
      yiN = di[1] - dN[1];
      ziN = di[2] - dN[2];
      riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

      double xjN=1.e-6, yjN=1.e-6, zjN=1.e-6, rjN=1.e-6;
      xjN = dj[0] - dN[0];
      yjN = dj[1] - dN[1];
      zjN = dj[2] - dN[2];
      rjN = pow(xjN*xjN + yjN*yjN + zjN*zjN, 0.5);
    
      double rijbar, dfij, d2fij;
      scaledRij(rij, rijbar, dfij, d2fij);
    
      double riNbar, dfiN, d2fiN;
      scaledRij(riN, riNbar, dfiN, d2fiN);

      double rjNbar, dfjN, d2fjN;
      scaledRij(rjN, rjNbar, dfjN, d2fjN);
      
      VectorXd iNPowers = VectorXd::Zero(maxQ+1), jNPowers = VectorXd::Zero(maxQ+1), ijPowers = VectorXd::Zero(maxQ+1);
      for (int m = 0; m <= maxQ; m++) {
        iNPowers[m] = pow(riNbar, m);
        jNPowers[m] = pow(rjNbar, m);
        ijPowers[m] = pow(rijbar, m);
      }

      for (int m = 1; m <= maxQ; m++) 
      for (int n = 0; n <= m   ; n++) 
      for (int o = 0; o <= (maxQ-m-n); o++) {
        if (n == 0 && o == 0) continue; //EN term

        double factor = params[startIndex + index];
        double value1 = iNPowers[m] * jNPowers[n] * ijPowers[o];
        double value2 = jNPowers[m] * iNPowers[n] * ijPowers[o];
      
        value   += factor*(value1 + value2);
        grad(0) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * xiN/riN
            + (o * value1 + o * value2) / rijbar * dfij * xij/rij );

        grad(1) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * yiN/riN
            + (o * value1 + o * value2) / rijbar * dfij * yij/rij );

        grad(2) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * ziN/riN
            + (o * value1 + o * value2) / rijbar * dfij * zij/rij );

        
        index++;
      }
    }
  }
  return value;
}



// (riN^m  jN^n + rjN^m riN^n) rij^o
// m + n + o <= maxQ
//returns an array of values, gx, gy, gz
double JastrowEENValue(int i, int j, int maxQ,
                       const vector<Vector3d>& r,
                       const VectorXd& params,
                       int startIndex,
                       int ss) {
  vector<double>& Ncharge = schd.Ncharge;
  double value = 0.0;
  int index = 0;
  
  for (int N=0; N<Ncharge.size(); N++)
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j], &dN = schd.Ncoords[N];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
      xiN = di[0] - dN[0];
      yiN = di[1] - dN[1];
      ziN = di[2] - dN[2];
      riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

      double xjN=1.e-6, yjN=1.e-6, zjN=1.e-6, rjN=1.e-6;
      xjN = dj[0] - dN[0];
      yjN = dj[1] - dN[1];
      zjN = dj[2] - dN[2];
      rjN = pow(xjN*xjN + yjN*yjN + zjN*zjN, 0.5);
    
      double rijbar, dfij, d2fij;
      scaledRij(rij, rijbar, dfij, d2fij);
    
      double riNbar, dfiN, d2fiN;
      scaledRij(riN, riNbar, dfiN, d2fiN);

      double rjNbar, dfjN, d2fjN;
      scaledRij(rjN, rjNbar, dfjN, d2fjN);

      VectorXd iNPowers = VectorXd::Zero(maxQ+1), jNPowers = VectorXd::Zero(maxQ+1), ijPowers = VectorXd::Zero(maxQ+1);
      for (int m = 0; m <= maxQ; m++) {
        iNPowers[m] = pow(riNbar, m);
        jNPowers[m] = pow(rjNbar, m);
        ijPowers[m] = pow(rijbar, m);
      }

      for (int m = 1; m <= maxQ; m++) 
      for (int n = 0; n <= m   ; n++) 
      for (int o = 0; o <= (maxQ-m-n); o++) {
        if (n == 0 && o == 0) continue; //EN term
        
        value += (iNPowers[m] * jNPowers[n]
                  + jNPowers[m] * iNPowers[n]) * ijPowers[o]
            * params[startIndex + index];
        index++;

      }
    }
  }
  return value;
}


// (riN^m  jN^n + rjN^m riN^n) rij^o
// m + n + o <= maxQ
//returns an array of values, gx, gy, gz
void JastrowEEN(int i, int j, int maxQ,
                const vector<Vector3d>& r,
                VectorXd& values, MatrixXd& gx,
                MatrixXd& gy, MatrixXd& gz,
                MatrixXd& laplace, double factor,
                int startIndex,
                int ss) {
  vector<double>& Ncharge = schd.Ncharge;
  int index = 0;
  
  for (int N=0; N<Ncharge.size(); N++)
  {
    if (electronsOfCorrectSpin(i, j, ss)) {
      
      const Vector3d& di = r[i], &dj = r[j], &dN = schd.Ncoords[N];
    
      double xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6;
      xij = di[0] - dj[0];
      yij = di[1] - dj[1];
      zij = di[2] - dj[2];
      rij = pow(xij*xij + yij*yij + zij*zij, 0.5);

      double xiN=1.e-6, yiN=1.e-6, ziN=1.e-6, riN=1.e-6;
      xiN = di[0] - dN[0];
      yiN = di[1] - dN[1];
      ziN = di[2] - dN[2];
      riN = pow(xiN*xiN + yiN*yiN + ziN*ziN, 0.5);

      double xjN=1.e-6, yjN=1.e-6, zjN=1.e-6, rjN=1.e-6;
      xjN = dj[0] - dN[0];
      yjN = dj[1] - dN[1];
      zjN = dj[2] - dN[2];
      rjN = pow(xjN*xjN + yjN*yjN + zjN*zjN, 0.5);
    
      double rijbar, dfij, d2fij;
      scaledRij(rij, rijbar, dfij, d2fij);
    
      double riNbar, dfiN, d2fiN;
      scaledRij(riN, riNbar, dfiN, d2fiN);

      double rjNbar, dfjN, d2fjN;
      scaledRij(rjN, rjNbar, dfjN, d2fjN);
      
      VectorXd iNPowers = VectorXd::Zero(maxQ+1), jNPowers = VectorXd::Zero(maxQ+1), ijPowers = VectorXd::Zero(maxQ+1);
      for (int m = 0; m <= maxQ; m++) {
        iNPowers[m] = pow(riNbar, m);
        jNPowers[m] = pow(rjNbar, m);
        ijPowers[m] = pow(rijbar, m);
      }

      double lapIntermediateI = (xiN * xij + yiN * yij + ziN * zij) / riN / rij;
      double lapIntermediateJ = (-xjN * xij - yjN * yij - zjN * zij) / rjN / rij;
      for (int m = 1; m <= maxQ; m++) 
      for (int n = 0; n <= m   ; n++) 
      for (int o = 0; o <= (maxQ-m-n); o++) {
        if (n == 0 && o == 0) continue; //EN term
        
        double value1 = iNPowers[m] * jNPowers[n] * ijPowers[o];
        double value2 = jNPowers[m] * iNPowers[n] * ijPowers[o];
      
        values[startIndex + index] += factor*(value1 + value2);

        gx(i, startIndex + index) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * xiN/riN
            + (o * value1 + o * value2) / rijbar * dfij * xij/rij );

        gy(i, startIndex + index) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * yiN/riN
            + (o * value1 + o * value2) / rijbar * dfij * yij/rij );

        gz(i, startIndex + index) += factor * (
              (m * value1 + n * value2) / riNbar * dfiN * ziN/riN
            + (o * value1 + o * value2) / rijbar * dfij * zij/rij );
        
        laplace(i, startIndex + index) += factor * (
              (m * value1 + n * value2) / riNbar * (d2fiN + dfiN * 2./riN)
            + (o * value1 + o * value2) / rijbar * (d2fij + dfij * 2/rij)
            + (m * (m - 1) * value1 + n * (n - 1) * value2) / riNbar / riNbar * dfiN * dfiN
            + (o * (o - 1) * (value1+value2)) / rijbar / rijbar * dfij * dfij
               
            + 2 * ( m  / riNbar * dfiN * value1 +  n  / riNbar * dfiN * value2)
                * (o / rijbar * dfij )
                * lapIntermediateI);
               

        gx(j, startIndex + index) += factor * (
            (n * value1 + m * value2) / rjNbar * dfjN * xjN/rjN
            - (o * value1 + o * value2) / rijbar * dfij * xij/rij);

        gy(j, startIndex + index) += factor * (
            (n * value1 + m * value2) / rjNbar * dfjN * yjN/rjN
            - (o * value1 + o * value2) / rijbar * dfij * yij/rij);

        gz(j, startIndex + index) += factor * (
            (n * value1 + m * value2) / rjNbar * dfjN * zjN/rjN
            - (o * value1 + o * value2) / rijbar * dfij * zij/rij);
        
        laplace(j, startIndex + index) += factor * (
              (n * value1 + m * value2) / rjNbar * (d2fjN + dfjN * 2./rjN)
            + (o * value1 + o * value2) / rijbar * (d2fij + dfij * 2/rij)
            + (n * (n - 1) * value1 + m * (m - 1) * value2) / rjNbar / rjNbar * dfjN * dfjN
            + (o * (o - 1) * (value1+value2)) / rijbar / rijbar * dfij * dfij
               
            + 2 * ( m  / rjNbar * dfjN * value2 +  n  / rjNbar * dfjN * value1)
                * (o / rijbar * dfij )
                * lapIntermediateJ);
        
        
        index++;
      }
    }
  }
}


void JastrowEENNinit(const vector<Vector3d> &r, VectorXd &N, MatrixXd &n, std::array<MatrixXd, 3> &gradn, MatrixXd &lapn)
{
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;

  int Nsize;
  if (schd.fourBodyJastrowBasis == NC) {
    Nsize = schd.Ncharge.size();
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    Nsize = norbs;
  }
  else if (schd.fourBodyJastrowBasis == SS) {
    Nsize = norbs;
  }
  
  N.setZero(2 * Nsize);
 
  n.setZero(nelec, Nsize);
  gradn[0].setZero(nelec, Nsize);
  gradn[1].setZero(nelec, Nsize);
  gradn[2].setZero(nelec, Nsize);                                
  lapn.setZero(nelec, Nsize);

  for (int i = 0; i < nelec; i++)
  {
    VectorXd pn;
    array<VectorXd, 3> gradpn, grad2pn;
    if (schd.fourBodyJastrowBasis == NC) {
      NC_eval_deriv2(i, r, pn, gradpn, grad2pn);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval_deriv2(i, r, pn, gradpn, grad2pn);
    }
    else if (schd.fourBodyJastrowBasis == SS) {
      SS_eval_deriv2(i, r, pn, gradpn, grad2pn);
    }

    n.row(i) = pn;

    gradn[0].row(i) = gradpn[0];
    gradn[1].row(i) = gradpn[1];
    gradn[2].row(i) = gradpn[2];

    lapn.row(i) = grad2pn[0] + grad2pn[1] + grad2pn[2];
  }

  for (int i = 0; i < Nsize; i++)
  {
    N[i] = n.col(i).head(nalpha).sum();
    N[i + Nsize] = n.col(i).tail(nbeta).sum();
  }
   
}

void JastrowEENN(const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, const MatrixXd &lapn, VectorXd &ParamValues, std::vector<MatrixXd> &ParamGradient, MatrixXd &ParamLaplacian, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  //linear term
  for (int I = 0; I < Nsize; I++)
  {
    int Ia = I;
    int Ib = I + Nsize;

    ParamValues[startIndex + Ia] = N[Ia];
    ParamValues[startIndex + Ib] = N[Ib];
    for (int i = 0; i < nelec; i++)
    {
      if (i < nalpha)
      {
        ParamGradient[0](i, startIndex + Ia) = gradn[0](i, I);
        ParamGradient[1](i, startIndex + Ia) = gradn[1](i, I);
        ParamGradient[2](i, startIndex + Ia) = gradn[2](i, I);
        ParamLaplacian(i, startIndex + Ia) = lapn(i, I);
      }
      else
      {
        ParamGradient[0](i, startIndex + Ib) = gradn[0](i, I);
        ParamGradient[1](i, startIndex + Ib) = gradn[1](i, I);
        ParamGradient[2](i, startIndex + Ib) = gradn[2](i, I);
        ParamLaplacian(i, startIndex + Ib) = lapn(i, I);
      }
    }
  }

  //quadratic term
  for (int I = 0; I < Nsize; I++)
  {
    for (int J = 0; J < Nsize; J++)
    {
      int Ia = I;
      int Ib = I + Nsize;
      int Ja = J;
      int Jb = J + Nsize;

      int stride = 2 * Nsize;

      int aaindex = startIndex + stride + Ia * stride + Ja;
      int bbindex = startIndex + stride + Ib * stride + Jb;
      int abindex = startIndex + stride + Ia * stride + Jb;
      int baindex = startIndex + stride + Ib * stride + Ja;

      ParamValues[aaindex] = N[Ia] * N[Ja];
      ParamValues[bbindex] = N[Ib] * N[Jb];
      ParamValues[abindex] = N[Ia] * N[Jb];
      ParamValues[baindex] = N[Ib] * N[Ja];
      
      for (int i = 0; i < nelec; i++)
      {
        if (i < nalpha)
        {
          ParamGradient[0](i, aaindex) = gradn[0](i, I) * N[Ja] + N[Ia] * gradn[0](i, J);
          ParamGradient[1](i, aaindex) = gradn[1](i, I) * N[Ja] + N[Ia] * gradn[1](i, J);
          ParamGradient[2](i, aaindex) = gradn[2](i, I) * N[Ja] + N[Ia] * gradn[2](i, J);

          ParamGradient[0](i, abindex) = gradn[0](i, I) * N[Jb];
          ParamGradient[1](i, abindex) = gradn[1](i, I) * N[Jb];
          ParamGradient[2](i, abindex) = gradn[2](i, I) * N[Jb];

          ParamGradient[0](i, baindex) = N[Ib] * gradn[0](i, J);
          ParamGradient[1](i, baindex) = N[Ib] * gradn[1](i, J);
          ParamGradient[2](i, baindex) = N[Ib] * gradn[2](i, J);

          ParamLaplacian(i, aaindex) = lapn(i, I) * N[Ja] + N[Ia] * lapn(i, J) + 2.0 * gradn[0](i, I) * gradn[0](i, J) 
                                                                               + 2.0 * gradn[1](i, I) * gradn[1](i, J) 
                                                                               + 2.0 * gradn[2](i, I) * gradn[2](i, J);

          ParamLaplacian(i, abindex) = lapn(i, I) * N[Jb];

          ParamLaplacian(i, baindex) = N[Ib] * lapn(i, J);
        }
        else
        {
          ParamGradient[0](i, bbindex) = gradn[0](i, I) * N[Jb] + N[Ib] * gradn[0](i, J);
          ParamGradient[1](i, bbindex) = gradn[1](i, I) * N[Jb] + N[Ib] * gradn[1](i, J);
          ParamGradient[2](i, bbindex) = gradn[2](i, I) * N[Jb] + N[Ib] * gradn[2](i, J);

          ParamGradient[0](i, abindex) = N[Ia] * gradn[0](i, J);
          ParamGradient[1](i, abindex) = N[Ia] * gradn[1](i, J);
          ParamGradient[2](i, abindex) = N[Ia] * gradn[2](i, J);

          ParamGradient[0](i, baindex) = gradn[0](i, I) * N[Ja];
          ParamGradient[1](i, baindex) = gradn[1](i, I) * N[Ja];
          ParamGradient[2](i, baindex) = gradn[2](i, I) * N[Ja];

          ParamLaplacian(i, bbindex) = lapn(i, I) * N[Jb] + N[Ib] * lapn(i, J) + 2.0 * gradn[0](i, I) * gradn[0](i, J) 
                                                                               + 2.0 * gradn[1](i, I) * gradn[1](i, J) 
                                                                               + 2.0 * gradn[2](i, I) * gradn[2](i, J);

          ParamLaplacian(i, abindex) = N[Ia] * lapn(i, J);

          ParamLaplacian(i, baindex) = lapn(i, I) * N[Ja];
        }
      }
    }
  }    
}

double JastrowEENNfactor(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, const VectorXd &params, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  Vector3d bkp = r[elec];
  const_cast<Vector3d&>(r[elec]) = coord;

  VectorXd nprime;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval(elec, r, nprime);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval(elec, r, nprime);
  }
  else if (schd.fourBodyJastrowBasis == SS) {
    SS_eval(elec, r, nprime);
  }

  const_cast<Vector3d&>(r[elec]) = bkp;

  //spin
  int sz = elec < nalpha ? 0 : 1;

  //N prime
  VectorXd Np = N;
  for (int i = 0; i < Nsize; i++)
  {
    int shift = sz * Nsize;
    Np(i + shift) += (nprime(i) - n(elec, i));
  }

  //linear term
  double val = 0;
  for (int I = 0; I < Nsize; I++)
  {
    int shift = sz * Nsize;
    double factor = params[startIndex + I + shift];
    val += factor * Np(I + shift);
    val -= factor * N(I + shift);
  }

  //quadratic term
  for (int I = 0; I < Nsize; I++)
  {
    for (int J = 0; J < Nsize; J++)
    {
      int Ia = I;
      int Ib = I + Nsize;
      int Ja = J;
      int Jb = J + Nsize;

      int stride = 2 * Nsize;
      if (sz = 0)
      {
        {
          int aaindex = startIndex + stride + Ia * stride + Ja;
          double factor = params[aaindex];
          val += factor * Np(Ia) * Np(Ja);
          val -= factor * N(Ia) * N(Ja);
        }
      }
      else
      {
        {
          int bbindex = startIndex + stride + Ib * stride + Jb;
          double factor = params[bbindex];
          val += factor * Np(Ib) * Np(Jb);
          val -= factor * N(Ib) * N(Jb);
        }
      }

      {
        int abindex = startIndex + stride + Ia * stride + Jb;
        double factor = params[abindex];
        val += factor * Np(Ia) * Np(Jb);
        val -= factor * N(Ia) * N(Jb);
      }

      {
        int baindex = startIndex + stride + Ib * stride + Ja;
        double factor = params[baindex];
        val += factor * Np(Ib) * Np(Ja);
        val -= factor * N(Ib) * N(Ja);
      }

    }
  }
  return val;
}

double JastrowEENNfactorAndGradient(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, Vector3d &grad, const VectorXd &params, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  Vector3d bkp = r[elec];
  const_cast<Vector3d&>(r[elec]) = coord;

  VectorXd np;
  array<VectorXd, 3> gradnp;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval_deriv(elec, r, np, gradnp);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval_deriv(elec, r, np, gradnp);
  }
  else if (schd.fourBodyJastrowBasis == SS) {
    SS_eval_deriv(elec, r, np, gradnp);
  }

  const_cast<Vector3d&>(r[elec]) = bkp;

  //spin
  int sz = elec < nalpha ? 0 : 1;

  //N prime
  VectorXd Np = N;
  for (int i = 0; i < Nsize; i++)
  {
    int shift = sz * Nsize;
    Np(i + shift) += (np(i) - n(elec, i));
  }

  //linear term
  double val = 0;
  for (int I = 0; I < Nsize; I++)
  {
    int shift = sz * Nsize;
    double factor = params[startIndex + I + shift];
    val += factor * Np(I + shift);
    val -= factor * N(I + shift);

    grad[0] += factor * (gradnp[0][I] - gradn[0](elec, I));
    grad[1] += factor * (gradnp[1][I] - gradn[1](elec, I));
    grad[2] += factor * (gradnp[2][I] - gradn[2](elec, I));
  }

  //quadratic term
  for (int I = 0; I < Nsize; I++)
  {
    for (int J = 0; J < Nsize; J++)
    {
      int Ia = I;
      int Ib = I + Nsize;
      int Ja = J;
      int Jb = J + Nsize;

      int stride = 2 * Nsize;
      if (sz = 0)
      {
        {
          int aaindex = startIndex + stride + Ia * stride + Ja;
          double factor = params[aaindex];

          val += factor * Np(Ia) * Np(Ja);
          val -= factor * N(Ia) * N(Ja);

          grad[0] += factor * (gradnp[0](I) * Np[Ja] + Np[Ia] * gradnp[0](J));
          grad[1] += factor * (gradnp[1](I) * Np[Ja] + Np[Ia] * gradnp[1](J));
          grad[2] += factor * (gradnp[2](I) * Np[Ja] + Np[Ia] * gradnp[2](J));

          grad[0] -= factor * (gradn[0](elec, I) * N[Ja] + N[Ia] * gradn[0](elec, J));
          grad[1] -= factor * (gradn[1](elec, I) * N[Ja] + N[Ia] * gradn[1](elec, J));
          grad[2] -= factor * (gradn[2](elec, I) * N[Ja] + N[Ia] * gradn[2](elec, J));
        }

        {
          int abindex = startIndex + stride + Ia * stride + Jb;
          double factor = params[abindex];

          val += factor * Np(Ia) * Np(Jb);
          val -= factor * N(Ia) * N(Jb);

          grad[0] += factor * gradnp[0](I) * Np[Jb];
          grad[1] += factor * gradnp[1](I) * Np[Jb];
          grad[2] += factor * gradnp[2](I) * Np[Jb];

          grad[0] -= factor * gradn[0](elec, I) * N[Jb];
          grad[1] -= factor * gradn[1](elec, I) * N[Jb];
          grad[2] -= factor * gradn[2](elec, I) * N[Jb];
        }

        {
          int baindex = startIndex + stride + Ib * stride + Ja;
          double factor = params[baindex];

          val += factor * Np(Ib) * Np(Ja);
          val -= factor * N(Ib) * N(Ja);

          grad[0] += factor * Np[Ib] * gradnp[0](J);
          grad[1] += factor * Np[Ib] * gradnp[1](J);
          grad[2] += factor * Np[Ib] * gradnp[2](J);

          grad[0] -= factor * N[Ib] * gradn[0](elec, J);
          grad[1] -= factor * N[Ib] * gradn[1](elec, J);
          grad[2] -= factor * N[Ib] * gradn[2](elec, J);
        }
      }
      else
      {
        {
          int bbindex = startIndex + stride + Ib * stride + Jb;
          double factor = params[bbindex];

          val += factor * Np(Ib) * Np(Jb);
          val -= factor * N(Ib) * N(Jb);

          grad[0] += factor * (gradnp[0](I) * Np[Jb] + Np[Ib] * gradnp[0](J));
          grad[1] += factor * (gradnp[1](I) * Np[Jb] + Np[Ib] * gradnp[1](J));
          grad[2] += factor * (gradnp[2](I) * Np[Jb] + Np[Ib] * gradnp[2](J));

          grad[0] -= factor * (gradn[0](elec, I) * N[Jb] + N[Ib] * gradn[0](elec, J));
          grad[1] -= factor * (gradn[1](elec, I) * N[Jb] + N[Ib] * gradn[1](elec, J));
          grad[2] -= factor * (gradn[2](elec, I) * N[Jb] + N[Ib] * gradn[2](elec, J));
        }

        {
          int abindex = startIndex + stride + Ia * stride + Jb;
          double factor = params[abindex];

          val += factor * Np(Ia) * Np(Jb);
          val -= factor * N(Ia) * N(Jb);

          grad[0] += factor * Np[Ia] * gradnp[0](J);
          grad[1] += factor * Np[Ia] * gradnp[1](J);
          grad[2] += factor * Np[Ia] * gradnp[2](J);

          grad[0] -= factor * N[Ia] * gradn[0](elec, J);
          grad[1] -= factor * N[Ia] * gradn[1](elec, J);
          grad[2] -= factor * N[Ia] * gradn[2](elec, J);
        }

        {
          int baindex = startIndex + stride + Ib * stride + Ja;
          double factor = params[baindex];

          val += factor * Np(Ib) * Np(Ja);
          val -= factor * N(Ib) * N(Ja);

          grad[0] += factor * gradnp[0](I) * Np[Ja];
          grad[1] += factor * gradnp[1](I) * Np[Ja];
          grad[2] += factor * gradnp[2](I) * Np[Ja];

          grad[0] -= factor * gradn[0](elec, I) * N[Ja];
          grad[1] -= factor * gradn[1](elec, I) * N[Ja];
          grad[2] -= factor * gradn[2](elec, I) * N[Ja];
        }
      }

    }
  }
  return val;
}

double JastrowEENNfactorVector(int elec, const Vector3d &coord, const vector<Vector3d> &r, const VectorXd &N, const MatrixXd &n, VectorXd &ParamValues, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  Vector3d bkp = r[elec];
  const_cast<Vector3d&>(r[elec]) = coord;

  VectorXd nprime;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval(elec, r, nprime);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval(elec, r, nprime);
  }
  else if (schd.fourBodyJastrowBasis == SS) {
    SS_eval(elec, r, nprime);
  }

  const_cast<Vector3d&>(r[elec]) = bkp;

  //spin
  int sz = elec < nalpha ? 0 : 1;

  //N prime
  VectorXd Np = N;
  for (int i = 0; i < Nsize; i++)
  {
    int shift = sz * Nsize;
    Np(i + shift) += (nprime(i) - n(elec, i));
  }

  //linear term
  for (int I = 0; I < Nsize; I++)
  {
    int Ia = I;
    int Ib = I + Nsize;

    ParamValues[startIndex + Ia] += Np[Ia];
    ParamValues[startIndex + Ia] -= N[Ia];

    ParamValues[startIndex + Ib] += Np[Ib];
    ParamValues[startIndex + Ib] -= N[Ib];
  }

  //quadratic term
  for (int I = 0; I < n.cols(); I++)
  {
    for (int J = 0; J < n.cols(); J++)
    {
      int Ia = I;
      int Ib = I + Nsize;
      int Ja = J;
      int Jb = J + Nsize;

      int stride = 2 * Nsize;

      int aaindex = startIndex + stride + Ia * stride + Ja;
      int bbindex = startIndex + stride + Ib * stride + Jb;
      int abindex = startIndex + stride + Ia * stride + Jb;
      int baindex = startIndex + stride + Ib * stride + Ja;

      ParamValues[aaindex] += Np[Ia] * Np[Ja];
      ParamValues[bbindex] += Np[Ib] * Np[Jb];
      ParamValues[abindex] += Np[Ia] * Np[Jb];
      ParamValues[baindex] += Np[Ib] * Np[Ja];

      ParamValues[aaindex] -= N[Ia] * N[Ja];
      ParamValues[bbindex] -= N[Ib] * N[Jb];
      ParamValues[abindex] -= N[Ia] * N[Jb];
      ParamValues[baindex] -= N[Ib] * N[Ja];
    }
  }
}

void JastrowEENNupdate(int elec, const Vector3d &coord, const vector<Vector3d> &r, VectorXd &N, MatrixXd &n, std::array<MatrixXd, 3> &gradn, MatrixXd &lapn, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  Vector3d bkp = r[elec];
  const_cast<Vector3d&>(r[elec]) = coord;

  VectorXd np;
  std::array<VectorXd, 3> gradnp, grad2np;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval_deriv2(elec, r, np, gradnp, grad2np);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval_deriv2(elec, r, np, gradnp, grad2np);
  }
  else if (schd.fourBodyJastrowBasis == SS) {
    SS_eval_deriv2(elec, r, np, gradnp, grad2np);
  }

  const_cast<Vector3d&>(r[elec]) = bkp;

  //spin
  int sz = elec < nalpha ? 0 : 1;

  //N prime
  VectorXd Np = N;
  for (int i = 0; i < Nsize; i++)
  {
    int shift = sz * Nsize;
    Np(i + shift) += (np(i) - n(elec, i));
  }

  N = Np;

  n.row(elec) = np;

  gradn[0].row(elec) = gradnp[0];
  gradn[1].row(elec) = gradnp[1];
  gradn[2].row(elec) = gradnp[2];

  lapn.row(elec) = grad2np[0] + grad2np[1] + grad2np[2];
}

void JastrowEENNupdateParam(int elec, const VectorXd &N, const MatrixXd &n, const std::array<MatrixXd, 3> &gradn, const MatrixXd &lapn, VectorXd &ParamValues, std::vector<MatrixXd> &ParamGradient, MatrixXd &ParamLaplacian, int startIndex)
{
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = rDeterminant::nalpha + rDeterminant::nbeta;
  int Nsize = n.cols();

  //spin
  int sz = elec < nalpha ? 0 : 1;

  //linear term
  for (int I = 0; I < Nsize; I++)
  {
    if (sz == 0) //alpha electron
    {
      int Ia = I;

      ParamValues[startIndex + Ia] = N[Ia];

      ParamGradient[0](elec, startIndex + Ia) = gradn[0](elec, I);
      ParamGradient[1](elec, startIndex + Ia) = gradn[1](elec, I);
      ParamGradient[2](elec, startIndex + Ia) = gradn[2](elec, I);

      ParamLaplacian(elec, startIndex + Ia) = lapn(elec, I);
    }
    else if (sz == 1) //beta electron
    {
      int Ib = I + Nsize;

      ParamValues[startIndex + Ib] = N[Ib];

      ParamGradient[0](elec, startIndex + Ib) = gradn[0](elec, I);
      ParamGradient[1](elec, startIndex + Ib) = gradn[1](elec, I);
      ParamGradient[2](elec, startIndex + Ib) = gradn[2](elec, I);

      ParamLaplacian(elec, startIndex + Ib) = lapn(elec, I);
    }
  }

  //quadratic term
  for (int I = 0; I < Nsize; I++)
  {
    for (int J = 0; J < Nsize; J++)
    {
      int Ia = I;
      int Ib = I + Nsize;
      int Ja = J;
      int Jb = J + Nsize;

      int stride = 2 * Nsize;

      int aaindex = startIndex + stride + Ia * stride + Ja;
      int bbindex = startIndex + stride + Ib * stride + Jb;
      int abindex = startIndex + stride + Ia * stride + Jb;
      int baindex = startIndex + stride + Ib * stride + Ja;

      if (sz == 0) //alpha electron
      {
        ParamValues[aaindex] = N[Ia] * N[Ja];
        ParamValues[abindex] = N[Ia] * N[Jb];
        ParamValues[baindex] = N[Ib] * N[Ja];

        for (int i = 0; i < nalpha; i++)
        {
          ParamGradient[0](i, aaindex) = gradn[0](i, I) * N[Ja] + N[Ia] * gradn[0](i, J);
          ParamGradient[1](i, aaindex) = gradn[1](i, I) * N[Ja] + N[Ia] * gradn[1](i, J);
          ParamGradient[2](i, aaindex) = gradn[2](i, I) * N[Ja] + N[Ia] * gradn[2](i, J);

          ParamLaplacian(i, aaindex) = lapn(i, I) * N[Ja] + N[Ia] * lapn(i, J) + 2.0 * gradn[0](i, I) * gradn[0](i, J) 
                                                                               + 2.0 * gradn[1](i, I) * gradn[1](i, J) 
                                                                               + 2.0 * gradn[2](i, I) * gradn[2](i, J);
        }

        for (int i = nalpha; i < nelec; i++)
        {
          ParamGradient[0](i, abindex) = N[Ia] * gradn[0](i, J);
          ParamGradient[1](i, abindex) = N[Ia] * gradn[1](i, J);
          ParamGradient[2](i, abindex) = N[Ia] * gradn[2](i, J);

          ParamGradient[0](i, baindex) = gradn[0](i, I) * N[Ja];
          ParamGradient[1](i, baindex) = gradn[1](i, I) * N[Ja];
          ParamGradient[2](i, baindex) = gradn[2](i, I) * N[Ja];

          ParamLaplacian(i, abindex) = N[Ia] * lapn(i, J);

          ParamLaplacian(i, baindex) = lapn(i, I) * N[Ja];
        }

        ParamGradient[0](elec, abindex) = gradn[0](elec, I) * N[Jb];
        ParamGradient[1](elec, abindex) = gradn[1](elec, I) * N[Jb];
        ParamGradient[2](elec, abindex) = gradn[2](elec, I) * N[Jb];

        ParamGradient[0](elec, baindex) = N[Ib] * gradn[0](elec, J);
        ParamGradient[1](elec, baindex) = N[Ib] * gradn[1](elec, J);
        ParamGradient[2](elec, baindex) = N[Ib] * gradn[2](elec, J);

        ParamLaplacian(elec, abindex) = lapn(elec, I) * N[Jb];

        ParamLaplacian(elec, baindex) = N[Ib] * lapn(elec, J);
      }
      else if (sz == 1) //beta electron
      {
        ParamValues[bbindex] = N[Ib] * N[Jb];
        ParamValues[abindex] = N[Ia] * N[Jb];
        ParamValues[baindex] = N[Ib] * N[Ja];

        for (int i = nalpha; i < nelec; i++)
        {
          ParamGradient[0](i, bbindex) = gradn[0](i, I) * N[Jb] + N[Ib] * gradn[0](i, J);
          ParamGradient[1](i, bbindex) = gradn[1](i, I) * N[Jb] + N[Ib] * gradn[1](i, J);
          ParamGradient[2](i, bbindex) = gradn[2](i, I) * N[Jb] + N[Ib] * gradn[2](i, J);
        
          ParamLaplacian(i, bbindex) = lapn(i, I) * N[Jb] + N[Ib] * lapn(i, J) + 2.0 * gradn[0](i, I) * gradn[0](i, J) 
                                                                               + 2.0 * gradn[1](i, I) * gradn[1](i, J) 
                                                                               + 2.0 * gradn[2](i, I) * gradn[2](i, J);
        }

        for (int i = 0; i < nalpha; i++)
        {
          ParamGradient[0](i, abindex) = gradn[0](i, I) * N[Jb];
          ParamGradient[1](i, abindex) = gradn[1](i, I) * N[Jb];
          ParamGradient[2](i, abindex) = gradn[2](i, I) * N[Jb];

          ParamGradient[0](i, baindex) = N[Ib] * gradn[0](i, J);
          ParamGradient[1](i, baindex) = N[Ib] * gradn[1](i, J);
          ParamGradient[2](i, baindex) = N[Ib] * gradn[2](i, J);

          ParamLaplacian(i, abindex) = lapn(i, I) * N[Jb];

          ParamLaplacian(i, baindex) = N[Ib] * lapn(i, J);
        }

        ParamGradient[0](elec, abindex) = N[Ia] * gradn[0](elec, J);
        ParamGradient[1](elec, abindex) = N[Ia] * gradn[1](elec, J);
        ParamGradient[2](elec, abindex) = N[Ia] * gradn[2](elec, J);

        ParamGradient[0](elec, baindex) = gradn[0](elec, I) * N[Ja];
        ParamGradient[1](elec, baindex) = gradn[1](elec, I) * N[Ja];
        ParamGradient[2](elec, baindex) = gradn[2](elec, I) * N[Ja];

        ParamLaplacian(elec, abindex) = N[Ia] * lapn(elec, J);

        ParamLaplacian(elec, baindex) = lapn(elec, I) * N[Ja];
      }
    }

  }

  /*
  if (schd.fourBodyJastrowBasis == NC || schd.fourBodyJastrowBasis == AB)
  {
  }
  else if (schd.fourBodyJastrowBasis == SS)
  {
  }
  */
}


double JastrowEENNLinearValue(int i, const vector<Vector3d> &r, const VectorXd &params, int startIndex)
{
  VectorXd n;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval(i, r, n);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval(i, r, n);
  }

  double value = 0.0;
  for (int I = 0; I < n.size(); I++)
  {
      value += params[startIndex + I] * n[I];
  }
  return value;
}


double JastrowEENNLinearValueGrad(int i, const vector<Vector3d> &r, Vector3d &grad, const VectorXd &params, int startIndex)
{
  VectorXd n;
  array<VectorXd, 3> gradn;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval_deriv(i, r, n, gradn);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval_deriv(i, r, n, gradn);
  }

  double value = 0.0;
  grad = Vector3d::Zero(3);
  for (int I = 0; I < n.size(); I++)
  {
      const double &factor = params[startIndex + I];
      value += factor * n[I];
      grad(0) += factor * gradn[0][I];
      grad(1) += factor * gradn[1][I];
      grad(2) += factor * gradn[2][I];
  }
  return value;
}


void JastrowEENNLinear(int i, const vector<Vector3d> &r, VectorXd &values, MatrixXd &gx, MatrixXd &gy, MatrixXd &gz, MatrixXd &laplace, double factor, int startIndex)
{
  VectorXd n;
  array<VectorXd, 3> gradn, grad2n;
  if (schd.fourBodyJastrowBasis == NC) {
    NC_eval_deriv2(i, r, n, gradn, grad2n);
  }
  else if (schd.fourBodyJastrowBasis == AB) {
    AB_eval_deriv2(i, r, n, gradn, grad2n);
  }

  for (int I = 0; I < n.size(); I++)
  {
    values[startIndex + I] += factor * n[I];
    gx(i, startIndex + I) += factor * gradn[0][I];
    gy(i, startIndex + I) += factor * gradn[1][I];
    gz(i, startIndex + I) += factor * gradn[2][I];
    laplace(i, startIndex + I) += factor * (grad2n[0][I] + grad2n[1][I] + grad2n[2][I]);
  }
}

double JastrowEENNValue(int i, int j, const vector<Vector3d> &r, const VectorXd &params, int startIndex)
{
  double value = 0.0;
  if (true) {

    VectorXd ni, nj;
    if (schd.fourBodyJastrowBasis == NC) {
      NC_eval(i, r, ni);
      NC_eval(j, r, nj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval(i, r, ni);
      AB_eval(j, r, nj);
    }

    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        value += params[startIndex + I * ni.size() + J] * ni[I] * nj[J];
      }
    }

  }
  return value;
}

double JastrowEENNValueGrad(int i, int j, const vector<Vector3d> &r, Vector3d &grad, const VectorXd &params, int startIndex)
{
  double value = 0.0;
  grad = Vector3d::Zero(3);
  if (true) {
    array<VectorXd, 3> gradni, gradnj;
    VectorXd ni, nj;

    if (schd.fourBodyJastrowBasis == NC) {
      NC_eval_deriv(i, r, ni, gradni);
      NC_eval_deriv(j, r, nj, gradnj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval_deriv(i, r, ni, gradni);
      AB_eval_deriv(j, r, nj, gradnj);
    }

    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        double factor = params[startIndex + I * ni.size() + J];
        value += factor * ni[I] * nj[J];
        if (i == j) {
          grad(0) += factor * (gradni[0][I] * nj[J] + ni[I] * gradnj[0][J]);
          grad(1) += factor * (gradni[1][I] * nj[J] + ni[I] * gradnj[1][J]);
          grad(2) += factor * (gradni[2][I] * nj[J] + ni[I] * gradnj[2][J]);
        }
        else {
          grad(0) += factor * gradni[0][I] * nj[J];
          grad(1) += factor * gradni[1][I] * nj[J];
          grad(2) += factor * gradni[2][I] * nj[J];
        }
      }
    }
  }
  return value;
}

void JastrowEENN(int i, int j, const vector<Vector3d> &r, VectorXd &values, MatrixXd &gx, MatrixXd &gy, MatrixXd &gz, MatrixXd &laplace, double factor, int startIndex)
{
  if (true) {
    VectorXd ni, nj;
    array<VectorXd, 3> gradni, gradnj, grad2ni, grad2nj;

    if (schd.fourBodyJastrowBasis == NC) {
      NC_eval_deriv2(i, r, ni, gradni, grad2ni);
      NC_eval_deriv2(j, r, nj, gradnj, grad2nj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval_deriv2(i, r, ni, gradni, grad2ni);
      AB_eval_deriv2(j, r, nj, gradnj, grad2nj);
    }

    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        values[startIndex + I * ni.size() + J] += factor * ni[I] * nj[J];

        gx(i, startIndex + I * ni.size() + J) += factor * gradni[0][I] * nj[J];
        gy(i, startIndex + I * ni.size() + J) += factor * gradni[1][I] * nj[J];
        gz(i, startIndex + I * ni.size() + J) += factor * gradni[2][I] * nj[J];
        laplace(i, startIndex + I * ni.size() + J) += factor * (grad2ni[0][I] + grad2ni[1][I] + grad2ni[2][I]) * nj[J];

        gx(j, startIndex + I * ni.size() + J) += factor * ni[I] * gradnj[0][J];
        gy(j, startIndex + I * ni.size() + J) += factor * ni[I] * gradnj[1][J];
        gz(j, startIndex + I * ni.size() + J) += factor * ni[I] * gradnj[2][J];
        laplace(j, startIndex + I * ni.size() + J) += factor * ni[I] * (grad2nj[0][J] + grad2nj[1][J] + grad2nj[2][J]);

        if (i == j)
        {
          laplace(i, startIndex + I * ni.size() + J) += factor * 2.0 * (gradni[0][I] * gradnj[0][J] + gradni[1][I] * gradnj[1][J] + gradni[2][I] * gradnj[2][J]);
        }
      }
    }
  }

}
