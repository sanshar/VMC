#include "slaterBasis.h"
#include <string>
#include "readSlater.h"

using namespace std;
using namespace Eigen;


double STOradialNorm(double ex, int n) {
  return pow(2*ex, 1.*n+0.5)/sqrt(tgamma(2*n));
}

double GTOradialNorm(double ex, int l) {
  double n = (2*l+2+1)*0.5;
  double integral = tgamma(n)/(2. * pow(2*ex, n));
  return 1./sqrt(integral);
}



void slaterBasis::read() {
  string fname = "slaterBasis.json";
  slaterParser sp(fname);

  sp.readBasis(atomList, *this);

  norbs = 0;
  //we will use cartesian basis
  for (int i=0; i<atomicBasis.size(); i++) {
    norbs += atomicBasis[i].norbs;
  }
  
}


int slaterBasis::getNorbs() {return norbs;}


void slaterBasis::eval(const Vector3d& x, double* values) {
  for (int i=0; i<atomicBasis.size(); i++) {
    auto aBasis = atomicBasis[i];

    double xN = x[0] - aBasis.coord[0],
        yN = x[1] - aBasis.coord[1],
        zN = x[2] - aBasis.coord[2];
    double RiN =  pow( pow(xN, 2) +
                       pow(yN, 2) +
                       pow(zN, 2), 0.5);


    int index = 0;
    //for each basis in aBasis evaluate the value
    for (int j=0; j<aBasis.exponents.size(); j++) {
      int l = aBasis.NL[2*j+1];
      int N = aBasis.NL[2*j]-1;
      double ex = pow(RiN, N) * aBasis.radialNorm[j] * exp(-aBasis.exponents[j]*RiN);
      
      if (l == 0) { //s
        values[index] = ex;
        index++;
      }
      if (l == 1) { //p
        values[index]    = xN * ex;
        values[index+1]  = yN * ex;
        values[index+2]  = zN * ex;
        index += 3;
      }
      if (l == 2) {//d
        values[index]    = xN * xN * ex;
        values[index+1]  = xN * yN * ex;
        values[index+3]  = xN * zN * ex;
        values[index+4]  = yN * yN * ex;
        values[index+5]  = yN * zN * ex;
        values[index+6]  = zN * zN * ex;
        index += 6;
      }
      if (l == 3) {//f
        values[index+0 ]    = xN * xN * xN * ex;
        values[index+1 ]    = xN * xN * yN * ex;
        values[index+2 ]    = xN * xN * zN * ex;
        values[index+3 ]    = xN * yN * yN * ex;
        values[index+4 ]    = xN * yN * zN * ex;
        values[index+5 ]    = xN * zN * zN * ex;
        values[index+6 ]    = yN * yN * yN * ex;
        values[index+7 ]    = yN * yN * zN * ex;
        values[index+8 ]    = yN * zN * zN * ex;
        values[index+9 ]    = zN * zN * zN * ex;
        index += 10;
      }
      
    }
  }

}



void slaterBasis::eval(const vector<Vector3d>& x, vector<double>& values) {
  int index = 0;
  for (int i=0; i<x.size(); i++) {
    eval(x[i], &values[index]);
    index += norbs;
  }
}

void makeDeriv(double* values, int NR, int NX, int NY, int NZ, int &norbs,
               double* xN, double &RiN, double& eta, double& scale,
               int& index) {

  double ex   = scale * exp(-eta*RiN);
  double poly = pow(RiN, NR);
  double ang  = pow(xN[0],NX) * pow (xN[1], NY) * pow (xN[2], NZ);
  double fr      = ex*poly;

  //*************************
  double DfDr   = -eta*ex*poly + NR*pow(RiN, NR-1)*ex;

  double DrDx = (xN[0]/RiN);
  double DrDy = (xN[1]/RiN);
  double DrDz = (xN[2]/RiN);

  double DangDx = abs(xN[0]) < 1.e-6 ?
      NX * pow(xN[0],NX-1) * pow (xN[1], NY) * pow (xN[2], NZ) :
      NX * ang / xN[0];

  double DangDy = abs(xN[1]) < 1.e-6 ?
      NY * pow(xN[0],NX) * pow (xN[1], NY-1) * pow (xN[2], NZ) :
      NY * ang / xN[1];

  double DangDz = abs(xN[2]) < 1.e-6 ?
      NZ * pow(xN[0],NX) * pow (xN[1], NY) * pow (xN[2], NZ-1) :
      NZ * ang / xN[2];

  //************************
  double D2fDr2 =  eta*eta*ex*poly
                    + 2 * (-eta*ex)*NR*pow(RiN, NR-1)
                    + NR*(NR-1)*pow(RiN, NR-2)*ex;

  double D2rDx2 = 1./RiN - xN[0]*xN[0]/pow(RiN,3);
  double D2rDy2 = 1./RiN - xN[1]*xN[1]/pow(RiN,3);
  double D2rDz2 = 1./RiN - xN[2]*xN[2]/pow(RiN,3);
  
  double D2angDx2 = abs(xN[0]) < 1.e-6 ?
      NX * (NX -1 ) * pow(xN[0],NX-2) * pow (xN[1], NY) * pow (xN[2], NZ) :
      NX * (NX -1 ) * ang / xN[0] /xN[0];

  double D2angDy2 = abs(xN[1]) < 1.e-6 ?
      NY * (NY -1 ) * pow(xN[0],NX) * pow (xN[1], NY-2) * pow (xN[2], NZ) :
      NY * (NY -1 ) * ang / xN[1] /xN[1];

  double D2angDz2 = abs(xN[2]) < 1.e-6 ?
      NZ * (NZ -1 ) * pow(xN[0],NX) * pow (xN[1], NY) * pow (xN[2], NZ-2) :
      NZ * (NZ -1 ) * ang / xN[2] /xN[2];

  //************************
  
  values[           index] =  fr * ang;
  values[1* norbs + index] =  DfDr * DrDx * ang + fr * DangDx;
  values[2* norbs + index] =  DfDr * DrDy * ang + fr * DangDy;
  values[3* norbs + index] =  DfDr * DrDz * ang + fr * DangDz;
  
  values[4* norbs + index] =  (D2fDr2*DrDx*DrDx + DfDr * D2rDx2)
                                 + 2 * (DfDr*DrDx) * DangDx
                                 + ex * D2angDx2;
  
  values[7* norbs + index] =  (D2fDr2*DrDy*DrDy + DfDr * D2rDy2)
                                 + 2 * (DfDr*DrDy) * DangDy
                                 + ex * D2angDy2;
  
  values[9* norbs + index] =  (D2fDr2*DrDz*DrDz + DfDr * D2rDz2)
                                 + 2 * (DfDr*DrDz) * DangDz
                                 + ex * D2angDz2;

}
  

void slaterBasis::eval_deriv2(const Vector3d& x, double* values) {
  for (int i=0; i<atomicBasis.size(); i++) {
    auto aBasis = atomicBasis[i];

    double xN[3];
    xN[0] = x[0] - aBasis.coord[0];
    xN[1] = x[1] - aBasis.coord[1];
    xN[2] = x[2] - aBasis.coord[2];
    
    double RiN =  pow( pow(xN[0], 2) +
                       pow(xN[1], 2) +
                       pow(xN[2], 2), 0.5);

    int index = 0;
    //for each basis in aBasis evaluate the value
    for (int j=0; j<aBasis.exponents.size(); j++) {
      int l = aBasis.NL[2*j+1];
      int N = aBasis.NL[2*j]-1;
      double eta   = aBasis.exponents[j];
      double scale = aBasis.radialNorm[j];
      
      if (l == 0) { //s
        makeDeriv(values, N, 0, 0, 0, norbs, xN, RiN, eta, scale, index);
        index++;
      }
      if (l == 1) { //p
        makeDeriv(values, N, 1, 0, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 1, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 0, 1, norbs, xN, RiN, eta, scale, index);
        index+=3;
      }
      if (l == 2) {//d
        makeDeriv(values, N, 2, 0, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 1, 1, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 1, 0, 1, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 2, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 1, 1, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 0, 2, norbs, xN, RiN, eta, scale, index);
        index+=6;
      }
      if (l == 3) {//f
        makeDeriv(values, N, 3, 0, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 2, 1, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 2, 0, 1, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 1, 2, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 1, 1, 1, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 1, 0, 2, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 3, 0, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 2, 1, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 1, 2, norbs, xN, RiN, eta, scale, index);
        makeDeriv(values, N, 0, 0, 3, norbs, xN, RiN, eta, scale, index);
        index+=10;        
      }                   
      
    }
  }

}
