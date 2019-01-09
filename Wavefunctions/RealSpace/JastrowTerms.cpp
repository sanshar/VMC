#include "JastrowTerms.h"
#include "input.h"
#include "global.h"


//fx = rbar1^n  rbar2^m rbar3^o
//rbar1 = rik/(1+beta rij)
//rbar2 = rjl/(1+beta ril)
//rbar3 = rij/(1+beta rjk)
double ExpGradLaplaceHelper(const int& n, const int& m, const int& o,
                            const double& scale, const double& beta,
                            const VectorXd& di, const VectorXd& dj,
                            const VectorXd& dk, const VectorXd& dl,
                            double &laplacei, double &laplacej,
                            double &gxi, double &gyi, double &gzi,
                            double &gxj, double &gyj, double &gzj,
                            double &gradHelper, double helperscale,
                            bool dolaplaceGrad) {

  //these cannot be zero because we divide by them to obtain derivatives
  double
      xij=1.e-6, yij=1.e-6, zij=1.e-6, rij=1.e-6,
      xik=1.e-6, yik=1.e-6, zik=1.e-6, rik=1.e-6,
      xjl=1.e-6, yjl=1.e-6, zjl=1.e-6, rjl=1.e-6;

  if (o != 0) {
    xij = di[0] - dj[0];
    yij = di[1] - dj[1];
    zij = di[2] - dj[2];
    rij = pow(xij*xij + yij*yij + zij*zij, 0.5);
  }
  
  if (n != 0) {
    xik = di[0] - dk[0];
    yik = di[1] - dk[1];
    zik = di[2] - dk[2];
    rik = pow(xik*xik + yik*yik + zik*zik, 0.5);
  }

  if (m != 0) {
    xjl = dj[0] - dl[0];
    yjl = dj[1] - dl[1];
    zjl = dj[2] - dl[2];
    rjl = pow(xjl*xjl + yjl*yjl + zjl*zjl, 0.5);
  }
  
  double rijbar = rij/(1. + beta*rij);
  double rikbar = rik/(1. + beta*rik);
  double rjlbar = rjl/(1. + beta*rjl);

  double value = pow(rikbar, n) * pow(rjlbar, m) * pow(rijbar, o);
  double f = scale * value;
  gradHelper += helperscale * value;
  
  if (dolaplaceGrad) {
    // grad and laplace w.r.t to xi
    double DfDrik = n == 0 ? 0 : n*f/rikbar * 1./pow(1+beta*rik, 2);
    double DfDrij = o == 0 ? 0 : o*f/rijbar * 1./pow(1+beta*rij, 2);

    gxi += DfDrik * xik/rik + DfDrij * xij/rij;
    gyi += DfDrik * yik/rik + DfDrij * yij/rij;
    gzi += DfDrik * zik/rik + DfDrij * zij/rij;

    if (n != 0)
    laplacei += 2 * DfDrik * (-beta/(1+beta*rik) + 1./rik)
        + n*(n-1)*f/rikbar/rikbar/pow(1.+beta*rik, 4);  //rik term

    if (o != 0)
    laplacei += 2 * DfDrij * (-beta/(1+beta*rij) + 1./rij)
        + o*(o-1)*f/rijbar/rijbar/pow(1.+beta*rij, 4);  //rij term

    if (o != 0 && n != 0)
    laplacei += 2*n*o*f/rikbar/rijbar
        /pow(1.+beta*rik, 2)/pow(1.+beta*rij, 2) *
        (xij*xik + yij*yik + zij*zik)/rij/rik;

  
    // grad and laplace w.r.t to xj
    double DfDrjl = m == 0 ? 0 : m*f/rjlbar * 1./pow(1+beta*rjl, 2);

    gxj += DfDrjl * xjl/rjl - DfDrij * xij/rij;
    gyj += DfDrjl * yjl/rjl - DfDrij * yij/rij;
    gzj += DfDrjl * zjl/rjl - DfDrij * zij/rij;

    if (m != 0)
    laplacej += 2 * DfDrjl * (-beta/(1+beta*rjl) + 1./rjl)
        + m*(m-1)*f/rjlbar/rjlbar/pow(1.+beta*rjl, 4);  //rjl term

    if (o != 0)
    laplacej += 2 * DfDrij * (-beta/(1+beta*rij) + 1./rij)
        + o*(o-1)*f/rijbar/rijbar/pow(1.+beta*rij, 4);  //rij term

    if (o != 0 && m != 0)
    laplacej += 2*m*o*f/rjlbar/rijbar
        /pow(1.+beta*rjl, 2)/pow(1.+beta*rij, 2) *
        (-xij*xjl - yij*yjl - zij*zjl)/rij/rjl;

  }
  return f;
}



GeneralJastrow::GeneralJastrow() : Ncoords(schd.Ncoords), Ncharge(schd.Ncharge) {
  beta = 1.0;
}

bool electronsOfCorrectSpin(const int& i, const int& j, const int& ss) {
  if ( ss == 2 ||  //applies to any term
       (i/rDeterminant::nalpha == j/rDeterminant::nalpha && ss == 1) ||
       (i/rDeterminant::nalpha != j/rDeterminant::nalpha && ss == 0) ) 
    return true;
  else
    return false;
}

double distance(const Vector3d& r1, const Vector3d& r2) {
  return pow( pow(r1[0] - r2[0], 2) +
              pow(r1[1] - r2[1], 2) +
              pow(r1[2] - r2[2], 2) , 0.5);
}

double GeneralJastrow::getExpLaplaceGradIJperJastrow(int i, int j,
                                                     vector<MatrixXd>& gradient,
                                                     MatrixXd& laplacian,
                                                     const Vector3d& coordi,
                                                     const Vector3d& coordj,
                                                     const double* params,
                                                     double * gradHelper,
                                                     double factor,
                                                     bool dolaplaceGrad) const {

  double exponent = 0.0;
  double laplacei, laplacej;
  double gix, giy, giz, gjx, gjy, gjz;
  for (int x = 0; x<I.size(); x++) {
    if (electronsOfCorrectSpin(i, j, ss[x])) {

      laplacei = 0; laplacej = 0;
      gix=0; giy=0; giz=0; gjx=0; gjy=0; gjz=0;
      exponent += ExpGradLaplaceHelper(m[x], n[x], o[x],
                                       params[x], beta,
                                       coordi, coordj,
                                       Ncoords[I[x]], Ncoords[J[x]],
                                       laplacei, laplacej,
                                       gix, giy, giz,
                                       gjx, gjy, gjz,
                                       gradHelper[x], factor, dolaplaceGrad);

      laplacian(i,x) += factor*laplacei; laplacian(j,x) += factor*laplacej;

      gradient[x](i,0) += factor*gix;
      gradient[x](i,1) += factor*giy;
      gradient[x](i,2) += factor*giz; 

      gradient[x](j,0) += factor*gjx;
      gradient[x](j,1) += factor*gjy;
      gradient[x](j,2) += factor*gjz; 
      
      if (n[x] != 0 || m[x] != 0) {
        laplacei = 0; laplacej = 0;
        gix=0; giy=0; giz=0; gjx=0; gjy=0; gjz=0;
        exponent += ExpGradLaplaceHelper(n[x], m[x], o[x],
                                         params[x], beta,
                                         coordi, coordj,
                                         Ncoords[J[x]], Ncoords[I[x]],
                                         laplacei, laplacej,
                                         gix, giy, giz,
                                         gjx, gjy, gjz,
                                         gradHelper[x], factor,
                                         dolaplaceGrad);
        laplacian(i,x) += factor*laplacei; laplacian(j,x) += factor*laplacej;

        gradient[x](i,0) += factor*gix;
        gradient[x](i,1) += factor*giy;
        gradient[x](i,2) += factor*giz; 
        
        gradient[x](j,0) += factor*gjx;
        gradient[x](j,1) += factor*gjy;
        gradient[x](j,2) += factor*gjz; 
        
      }
    }
  }

  return exponent;
}


double GeneralJastrow::getExpLaplaceGradIJ(int i, int j,
                                           Vector3d& gi, Vector3d& gj,
                                           double& laplaciani, double& laplacianj,
                                           const Vector3d& coordi,
                                           const Vector3d& coordj,
                                           const double* params,
                                           double * gradHelper,
                                           double factor,
                                           bool dolaplaceGrad) const {

  double exponent = 0.0;
  for (int x = 0; x<I.size(); x++) {
    if (electronsOfCorrectSpin(i, j, ss[x])) {

      exponent += ExpGradLaplaceHelper(m[x], n[x], o[x],
                                       params[x], beta,
                                       coordi, coordj,
                                       Ncoords[I[x]], Ncoords[J[x]],
                                       laplaciani, laplacianj,
                                       gi[0], gi[1], gi[2],
                                       gj[0], gj[1], gj[2],
                                       gradHelper[x], factor, dolaplaceGrad);

      if (n[x] != 0 || m[x] != 0) {
        exponent += ExpGradLaplaceHelper(n[x], m[x], o[x],
                                         params[x], beta,
                                         coordi, coordj,
                                         Ncoords[J[x]], Ncoords[I[x]],
                                         laplaciani, laplacianj,
                                         gi[0], gi[1], gi[2],
                                         gj[0], gj[1], gj[2],
                                         gradHelper[x], factor,
                                         dolaplaceGrad);
      }
    }
  }

  return exponent;
}


double GeneralJastrow::exponentialInitLaplaceGrad(const rDeterminant& d,
                                                  vector<MatrixXd>& ParamGradient,
                                                  MatrixXd& ParamLaplacian,
                                                  const double * params,
                                                  double * gradHelper) const {

  int nalpha = rDeterminant::nalpha;

  double exponent = 0.0;
  for (int i=0; i<d.nelec; i++)
    for (int j=0; j<i; j++) {
      exponent += getExpLaplaceGradIJperJastrow(i, j, ParamGradient,
                                                ParamLaplacian,
                                                d.coord[i], d.coord[j],
                                                params, gradHelper, 1.0, true);
    }


  return exponent;  
}

double GeneralJastrow::exponentDiff(int i, const Vector3d &newcoord,
                                    const rDeterminant &d,
                                    const double * params,
                                    double * gradHelper) const
{
  Vector3d gi, gj;
  double laplacei, laplacej;
  
  double diff = 0.0;
  for (int j=0; j<d.nelec; j++) {
    if (i == j) continue;

    diff += getExpLaplaceGradIJ(i, j, gi, gj, laplacei, laplacej,
                                newcoord, d.coord[j],
                                params, gradHelper, 0.0, false);
    
    diff -= getExpLaplaceGradIJ(i, j, gi, gj, laplacei, laplacej,
                                d.coord[i], d.coord[j],
                                params, gradHelper, 0.0, false);
  }

  return diff;
}



void GeneralJastrow::UpdateLaplaceGrad(vector<MatrixXd>& Gradient,
                                       MatrixXd& laplacian,
                                       const rDeterminant& d,
                                       const Vector3d& oldCoord,
                                       int i,
                                       const double * params,
                                       double * gradHelper) const {
  
  for (int j=0; j<d.nelec; j++) {
    if (j == i) continue;

    getExpLaplaceGradIJperJastrow(i, j, Gradient,
                                  laplacian,
                                  d.coord[i], d.coord[j],
                                  params, gradHelper, 1.0, true);
    

    getExpLaplaceGradIJperJastrow(i, j, Gradient,
                                  laplacian,
                                  oldCoord, d.coord[j],
                                  params, gradHelper, -1.0, true);
    
  }
}


void GeneralJastrow::OverlapWithGradient(VectorXd& grad, int& index,
                                         const rDeterminant& d,
                                         const vector<double>& params,
                                         const vector<double>& gradHelper) const {

  for (int i=0; i<gradHelper.size(); i++) {

    if (fixed[i] == 0) {
      grad[index] = gradHelper[i]; index++;
    }
    else {
      grad[index] = 0; index++;
    }

  }

}












