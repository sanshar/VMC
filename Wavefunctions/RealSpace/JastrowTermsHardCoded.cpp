#include "JastrowTermsHardCoded.h"
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
  //for (int i=0; i<r.size(); i++)
  //for (int j=0; j<i; j++)
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
void JastrowEE(int i, int j, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex,
               int ss) {

  
  //for (int i=0; i<r.size(); i++)
  //for (int j=0; j<i; j++)
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
  
  //for (int i=0; i<r.size(); i++)
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

//input take the maximum exponent Q (rij^o). o = 1...maxQ
//returns an array of values, gx, gy, gz
void JastrowEN(int i, int maxQ,
               const vector<Vector3d>& r,
               VectorXd& values, MatrixXd& gx,
               MatrixXd& gy, MatrixXd& gz,
               MatrixXd& laplace, double factor,
               int startIndex) {

  vector<double>& Ncharge = schd.Ncharge;
  
  //for (int i=0; i<r.size(); i++)
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
  
  //for (int i=0; i<r.size(); i++)
  //for (int j=0; j<i; j++)
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

      for (int m = 1; m <= maxQ; m++) 
      for (int n = 0; n <= m   ; n++) 
      for (int o = 0; o <= (maxQ-m-n); o++) {
        if (n == 0 && o == 0) continue; //EN term
        
        value += (pow(riNbar, m) * pow(rjNbar, n)
                  + pow(rjNbar, m) * pow(riNbar, n)) * pow(rijbar, o)
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
  
  //for (int i=0; i<r.size(); i++)
  //for (int j=0; j<i; j++)
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

      for (int m = 1; m <= maxQ; m++) 
      for (int n = 0; n <= m   ; n++) 
      for (int o = 0; o <= (maxQ-m-n); o++) {
        if (n == 0 && o == 0) continue; //EN term
        
        double value1 = pow(riNbar, m) * pow(rjNbar, n) * pow(rijbar, o);
        double value2 = pow(rjNbar, m) * pow(riNbar, n) * pow(rijbar, o);
      
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
                * (xiN * xij + yiN * yij + ziN * zij) / riN / rij);
               

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
                * (-xjN * xij - yjN * yij - zjN * zij) / rjN / rij);
        
        
        index++;
      }
    }
  }
}














