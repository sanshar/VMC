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


double JastrowENValueGrad(int i, int maxQ,
                          const vector<Vector3d>& r,
                          Vector3d& grad,
                          const VectorXd& params,
                          int startIndex) {

  double value = 0.0;
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


double JastrowEENNLinearValue(int i, const vector<Vector3d>& r, const VectorXd& params, int startIndex)
{
  /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);
  schd.basis->eval(r[i], &aoValues[0]);

  VectorXd n = VectorXd::Zero(Ncharge.size());
  double denom = 0.0;
  int orb = 0;
  for (int I = 0; I < n.size(); I++)
  {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          n[I] += aoValues[orb] * aoValues[orb];
          denom += aoValues[orb] * aoValues[orb];
          orb++;
      }
  }
  n /= denom;
  //cout << n.transpose() << endl << endl;
  */

  VectorXd n;
  if (schd.fourBodyJastrowBasis == FC) {
    FC_eval(i, r, n);
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

double JastrowEENNLinearValueGrad(int i, const vector<Vector3d>& r, Vector3d& grad, const VectorXd& params, int startIndex)
{
  /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);
  schd.basis->eval_deriv2(r[i], &aoValues[0]);

  VectorXd n = VectorXd::Zero(Ncharge.size());
  VectorXd num = VectorXd::Zero(Ncharge.size());
  VectorXd gradxNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradyNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradzNum = VectorXd::Zero(Ncharge.size());
  double denom = 0.0;
  Vector3d gradDenom = Vector3d::Zero(3);
  int orb = 0;
  for (int I = 0; I < n.size(); I++)
  {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          num[I] += aoValues[orb] * aoValues[orb];
          denom += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNum[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          //grady
          gradyNum[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          //gradz
          gradzNum[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          orb++;
      }
  }
  n = num / denom;
  VectorXd gradxn = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
  VectorXd gradyn = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
  VectorXd gradzn = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);


  double value = 0.0;
  grad = Vector3d::Zero(3);
  for (int I = 0; I < Ncharge.size(); I++)
  {
      const double &factor = params[startIndex + I];
      value += factor * n[I];
      grad(0) += factor * gradxn[I];
      grad(1) += factor * gradyn[I];
      grad(2) += factor * gradzn[I];
  }
  return value;
  */

  VectorXd n;
  array<VectorXd, 3> gradn;
  if (schd.fourBodyJastrowBasis == FC) {
    FC_eval_deriv(i, r, n, gradn);
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

void JastrowEENNLinear(int i, const vector<Vector3d>& r, VectorXd& values, MatrixXd& gx, MatrixXd& gy, MatrixXd& gz, MatrixXd& laplace, double factor, int startIndex)
{
    /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);
  schd.basis->eval_deriv2(r[i], &aoValues[0]);

  VectorXd n = VectorXd::Zero(Ncharge.size());
  VectorXd num = VectorXd::Zero(Ncharge.size());
  VectorXd gradxNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradyNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradzNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradxxNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradyyNum = VectorXd::Zero(Ncharge.size());
  VectorXd gradzzNum = VectorXd::Zero(Ncharge.size());
  double denom = 0.0;
  Vector3d gradDenom = Vector3d::Zero(3);
  Vector3d grad2Denom = Vector3d::Zero(3);
  int orb = 0;
  for (int I = 0; I < n.size(); I++)
  {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          num[I] += aoValues[orb] * aoValues[orb];
          denom += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNum[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradxxNum[I] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];
          grad2Denom[0] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];

          //grady
          gradyNum[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradyyNum[I] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];
          grad2Denom[1] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];

          //gradz
          gradzNum[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradzzNum[I] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];
          grad2Denom[2] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];

          orb++;
      }
  }
  n = num / denom;
  VectorXd gradxn = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
  VectorXd gradyn = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
  VectorXd gradzn = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);

  VectorXd gradxxn = (denom * denom * gradxxNum
                    - denom * (2.0 * gradxNum * gradDenom[0] + num * grad2Denom[0])
                    + 2.0 * num * gradDenom[0] * gradDenom[0]) / (denom * denom * denom);
  VectorXd gradyyn = (denom * denom * gradyyNum
                    - denom * (2.0 * gradyNum * gradDenom[1] + num * grad2Denom[1])
                    + 2.0 * num * gradDenom[1] * gradDenom[1]) / (denom * denom * denom);
  VectorXd gradzzn = (denom * denom * gradzzNum
                    - denom * (2.0 * gradzNum * gradDenom[2] + num * grad2Denom[2])
                    + 2.0 * num * gradDenom[2] * gradDenom[2]) / (denom * denom * denom);

  for (int I = 0; I < Ncharge.size(); I++)
  {
      values[startIndex + I] += factor * n[I];
      gx(i, startIndex + I) += factor * gradxn[I];
      gy(i, startIndex + I) += factor * gradyn[I];
      gz(i, startIndex + I) += factor * gradzn[I];
      laplace(i, startIndex + I) += factor * (gradxxn[I] + gradyyn[I] + gradzzn[I]);
  }
*/
  VectorXd n;
  array<VectorXd, 3> gradn, grad2n;
  if (schd.fourBodyJastrowBasis == FC) {
    FC_eval_deriv2(i, r, n, gradn, grad2n);
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
  /*
  cout << "EENNlinear" << endl;
  cout <<  i << " " << n.transpose() << endl;
  cout << endl;
  */
}

double JastrowEENNValue(int i, int j, const vector<Vector3d>& r, const VectorXd& params, int startIndex, int ss)
{
  /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;
  double value = 0.0;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);

  if (electronsOfCorrectSpin(i, j, ss)) {
    //cout << "JastrowEENNValue" << endl;
    VectorXd ni = VectorXd::Zero(Ncharge.size());
    schd.basis->eval(r[i], &aoValues[0]);
    double denom = 0.0;
    int orb = 0;
    for (int I = 0; I < ni.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          ni[I] += aoValues[orb] * aoValues[orb];
          denom += aoValues[orb] * aoValues[orb];
          orb++;
      }
    }
    ni /= denom;
    //cout << ni.transpose() << endl;

    VectorXd nj = VectorXd::Zero(Ncharge.size());
    schd.basis->eval(r[j], &aoValues[0]);
    denom = 0.0;
    orb = 0;
    for (int I = 0; I < nj.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          nj[I] += aoValues[orb] * aoValues[orb];
          denom += aoValues[orb] * aoValues[orb];
          orb++;
      }
    }
    nj /= denom;
    //cout << nj.transpose() << endl << endl;
    */

  double value = 0.0;
  if (electronsOfCorrectSpin(i, j, ss)) {

    VectorXd ni, nj;
    if (schd.fourBodyJastrowBasis == FC) {
      FC_eval(i, r, ni);
      FC_eval(j, r, nj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval(i, r, ni);
      AB_eval(j, r, nj);
    }

    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        int _I = std::max(I, J);
        int _J = std::min(I, J);
        value += params[startIndex + _I * (_I + 1) / 2 + _J] * ni[I] * nj[J];
      }
    }
  }
  return value;
}

double JastrowEENNValueGrad(int i, int j, const vector<Vector3d>& r, Vector3d grad, const VectorXd& params, int startIndex, int ss)
{
  /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);

  if (electronsOfCorrectSpin(i, j, ss)) {
    //cout << "JastrowEENNValueGrad" << endl;
    VectorXd ni = VectorXd::Zero(Ncharge.size());
    VectorXd numi = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNumi = VectorXd::Zero(Ncharge.size());
    double denomi = 0.0;
    Vector3d gradDenomi = Vector3d::Zero(3);
    schd.basis->eval_deriv2(r[i], &aoValues[0]);
    int orb = 0;
    for (int I = 0; I < ni.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          numi[I] += aoValues[orb] * aoValues[orb];
          denomi += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNumi[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenomi[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          //grady
          gradyNumi[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenomi[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          //gradz
          gradzNumi[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenomi[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          orb++;
      }
    }
    ni = numi / denomi;
    //cout << ni.transpose() << endl;
    VectorXd gradxni = (gradxNumi * denomi - numi * gradDenomi[0]) / (denomi * denomi);
    VectorXd gradyni = (gradyNumi * denomi - numi * gradDenomi[1]) / (denomi * denomi);
    VectorXd gradzni = (gradzNumi * denomi - numi * gradDenomi[2]) / (denomi * denomi);

    VectorXd nj = VectorXd::Zero(Ncharge.size());
    VectorXd numj = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNumj = VectorXd::Zero(Ncharge.size());
    double denomj = 0.0;
    Vector3d gradDenomj = Vector3d::Zero(3);
    schd.basis->eval_deriv2(r[j], &aoValues[0]);
    orb = 0;
    for (int I = 0; I < ni.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          numj[I] += aoValues[orb] * aoValues[orb];
          denomj += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNumj[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenomj[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          //grady
          gradyNumj[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenomj[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          //gradz
          gradzNumj[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenomj[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          orb++;
      }
    }
    nj = numj / denomj;
    //cout << nj.transpose() << endl;
    VectorXd gradxnj = (gradxNumj * denomj - numj * gradDenomj[0]) / (denomj * denomj);
    VectorXd gradynj = (gradyNumj * denomj - numj * gradDenomj[1]) / (denomj * denomj);
    VectorXd gradznj = (gradzNumj * denomj - numj * gradDenomj[2]) / (denomj * denomj);
    */

  double value = 0.0;
  grad = Vector3d::Zero(3);
  if (electronsOfCorrectSpin(i, j, ss)) {

    array<VectorXd, 3> gradni, gradnj;
    VectorXd ni, nj;

    if (schd.fourBodyJastrowBasis == FC) {
      FC_eval_deriv(i, r, ni, gradni);
      FC_eval_deriv(j, r, nj, gradnj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval_deriv(i, r, ni, gradni);
      AB_eval_deriv(j, r, nj, gradnj);
    }

    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        int _I = std::max(I, J);
        int _J = std::min(I, J);

        double factor = params[startIndex + _I * (_I + 1) / 2 + _J];
        value += factor * ni[I] * nj[J];
        grad(0) += factor * (gradni[0][I] * nj[J]);
        grad(1) += factor * (gradni[1][I] * nj[J]);
        grad(2) += factor * (gradni[2][I] * nj[J]);
      }
    }
  }
  return value;
}

void JastrowEENN(int i, int j, const vector<Vector3d>& r, VectorXd& values, MatrixXd& gx, MatrixXd& gy, MatrixXd& gz, MatrixXd& laplace, double factor, int startIndex, int ss)
{
  /*
  vector<double>& Ncharge = schd.Ncharge;
  vector<int>& Nbasis = schd.Nbasis;
  vector<Vector3d>& Ncoords = schd.Ncoords;

  int norbs = schd.basis->getNorbs();
  vector<double> aoValues(10*norbs, 0.0);

  if (electronsOfCorrectSpin(i, j, ss)) {
    //cout << "JastrowEENN" << endl;
    schd.basis->eval_deriv2(r[i], &aoValues[0]);
    VectorXd ni = VectorXd::Zero(Ncharge.size());
    VectorXd numi = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradxxNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradyyNumi = VectorXd::Zero(Ncharge.size());
    VectorXd gradzzNumi = VectorXd::Zero(Ncharge.size());
    double denomi = 0.0;
    Vector3d gradDenomi = Vector3d::Zero(3);
    Vector3d grad2Denomi = Vector3d::Zero(3);
    int orb = 0;
    for (int I = 0; I < ni.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          numi[I] += aoValues[orb] * aoValues[orb];
          denomi += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNumi[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenomi[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradxxNumi[I] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];
          grad2Denomi[0] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];

          //grady
          gradyNumi[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenomi[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradyyNumi[I] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];
          grad2Denomi[1] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];

          //gradz
          gradzNumi[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenomi[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradzzNumi[I] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];
          grad2Denomi[2] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];

          orb++;
      }
    }
    ni = numi / denomi;
    //cout << ni.transpose() << endl;
    VectorXd gradxni = (gradxNumi * denomi - numi * gradDenomi[0]) / (denomi * denomi);
    VectorXd gradyni = (gradyNumi * denomi - numi * gradDenomi[1]) / (denomi * denomi);
    VectorXd gradzni = (gradzNumi * denomi - numi * gradDenomi[2]) / (denomi * denomi);

    VectorXd gradxxni = (denomi * denomi * gradxxNumi
                      - denomi * (2.0 * gradxNumi * gradDenomi[0] + numi * grad2Denomi[0])
                      + 2.0 * numi * gradDenomi[0] * gradDenomi[0]) / (denomi * denomi * denomi);
    VectorXd gradyyni = (denomi * denomi * gradyyNumi
                      - denomi * (2.0 * gradyNumi * gradDenomi[1] + numi * grad2Denomi[1])
                      + 2.0 * numi * gradDenomi[1] * gradDenomi[1]) / (denomi * denomi * denomi);
    VectorXd gradzzni = (denomi * denomi * gradzzNumi
                      - denomi * (2.0 * gradzNumi * gradDenomi[2] + numi * grad2Denomi[2])
                      + 2.0 * numi * gradDenomi[2] * gradDenomi[2]) / (denomi * denomi * denomi);

    schd.basis->eval_deriv2(r[j], &aoValues[0]);
    VectorXd nj = VectorXd::Zero(Ncharge.size());
    VectorXd numj = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradxxNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradyyNumj = VectorXd::Zero(Ncharge.size());
    VectorXd gradzzNumj = VectorXd::Zero(Ncharge.size());
    double denomj = 0.0;
    Vector3d gradDenomj = Vector3d::Zero(3);
    Vector3d grad2Denomj = Vector3d::Zero(3);
    orb = 0;
    for (int I = 0; I < nj.size(); I++)
    {
      for (int mu = 0; mu < Nbasis[I]; mu++)
      {
          numj[I] += aoValues[orb] * aoValues[orb];
          denomj += aoValues[orb] * aoValues[orb];
          
          //gradx
          gradxNumj[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradDenomj[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
          gradxxNumj[I] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];
          grad2Denomj[0] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];

          //grady
          gradyNumj[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradDenomj[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
          gradyyNumj[I] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];
          grad2Denomj[1] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];

          //gradz
          gradzNumj[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradDenomj[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
          gradzzNumj[I] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];
          grad2Denomj[2] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];

          orb++;
      }
    }
    nj = numj / denomj;
    //cout << nj.transpose() << endl;
    VectorXd gradxnj = (gradxNumj * denomj - numj * gradDenomj[0]) / (denomj * denomj);
    VectorXd gradynj = (gradyNumj * denomj - numj * gradDenomj[1]) / (denomj * denomj);
    VectorXd gradznj = (gradzNumj * denomj - numj * gradDenomj[2]) / (denomj * denomj);

    VectorXd gradxxnj = (denomj * denomj * gradxxNumj
                      - denomj * (2.0 * gradxNumj * gradDenomj[0] + numj * grad2Denomj[0])
                      + 2.0 * numj * gradDenomj[0] * gradDenomj[0]) / (denomj * denomj * denomj);
    VectorXd gradyynj = (denomj * denomj * gradyyNumj
                      - denomj * (2.0 * gradyNumj * gradDenomj[1] + numj * grad2Denomj[1])
                      + 2.0 * numj * gradDenomj[1] * gradDenomj[1]) / (denomj * denomj * denomj);
    VectorXd gradzznj = (denomj * denomj * gradzzNumj
                      - denomj * (2.0 * gradzNumj * gradDenomj[2] + numj * grad2Denomj[2])
                      + 2.0 * numj * gradDenomj[2] * gradDenomj[2]) / (denomj * denomj * denomj);

    */

  if (electronsOfCorrectSpin(i, j, ss)) {

    VectorXd ni, nj;
    array<VectorXd, 3> gradni, gradnj, grad2ni, grad2nj;

    if (schd.fourBodyJastrowBasis == FC) {
      FC_eval_deriv2(i, r, ni, gradni, grad2ni);
      FC_eval_deriv2(j, r, nj, gradnj, grad2nj);
    }
    else if (schd.fourBodyJastrowBasis == AB) {
      AB_eval_deriv2(i, r, ni, gradni, grad2ni);
      AB_eval_deriv2(j, r, nj, gradnj, grad2nj);
    }


    for (int I = 0; I < ni.size(); I++)
    {
      for (int J = 0; J < nj.size(); J++)
      {
        int _I = std::max(I, J);
        int _J = std::min(I, J);

        values[startIndex + _I * (_I + 1) / 2 + _J] += factor * ni[I] * nj[J];

        gx(i, startIndex + _I * (_I + 1) / 2 + _J) += factor * gradni[0][I] * nj[J];
        gy(i, startIndex + _I * (_I + 1) / 2 + _J) += factor * gradni[1][I] * nj[J];
        gz(i, startIndex + _I * (_I + 1) / 2 + _J) += factor * gradni[2][I] * nj[J];
        laplace(i, startIndex + _I * (_I + 1) / 2 + _J) += factor * (grad2ni[0][I] * nj[J]
                                                                   + grad2ni[1][I] * nj[J]
                                                                   + grad2ni[2][I] * nj[J]);

        gx(j, startIndex + _I * (_I + 1) / 2 + _J) += factor * ni[I] * gradnj[0][J];
        gy(j, startIndex + _I * (_I + 1) / 2 + _J) += factor * ni[I] * gradnj[1][J];
        gz(j, startIndex + _I * (_I + 1) / 2 + _J) += factor * ni[I] * gradnj[2][J];
        laplace(j, startIndex + _I * (_I + 1) / 2 + _J) += factor * (ni[I] * grad2nj[0][J]
                                                                   + ni[I] * grad2nj[1][J]
                                                                   + ni[I] * grad2nj[2][J]);
      }
    }

    /*
    if (i == 0 && j == 1) {
    cout << "EENN basis" << endl;
    cout << r[0].transpose() << endl;
    cout << r[1].transpose() << endl << endl;
    cout << ni[0] << " " << nj[0] << " | " << ni[0] + nj[0] << endl;
    cout << ni[1] << " " << nj[1] << " | " << ni[1] + nj[1] << endl;
    cout << endl;
    cout << i << endl;
    cout << gradni[0].transpose() << endl;
    cout << gradni[1].transpose() << endl;
    cout << gradni[2].transpose() << endl << endl;
    cout << grad2ni[0].transpose() << endl;
    cout << grad2ni[1].transpose() << endl;
    cout << grad2ni[2].transpose() << endl << endl;
    cout << endl;
    cout << j << endl;
    cout << gradnj[0].transpose() << endl;
    cout << gradnj[1].transpose() << endl;
    cout << gradnj[2].transpose() << endl << endl;
    cout << grad2nj[0].transpose() << endl;
    cout << grad2nj[1].transpose() << endl;
    cout << grad2nj[2].transpose() << endl << endl;
    }
    */
  }
}

