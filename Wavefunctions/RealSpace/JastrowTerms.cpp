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

bool electronsOfCorrectSpin(int i, int j, int ss) {
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

double GeneralJastrow::getExpLaplaceGradIJ(int i, int j,
                                           Vector3d& gi, Vector3d& gj,
                                           double& laplaciani,
                                           double& laplacianj,
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
                                                  MatrixXd& Gradient,
                                                  VectorXd& laplacian,
                                                  const double * params,
                                                  double * gradHelper) const {

  int nalpha = rDeterminant::nalpha;
  Vector3d gi, gj;
  
  double exponent = 0.0;
  for (int i=0; i<d.nelec; i++)
    for (int j=0; j<i; j++) {
      gi.setZero(); gj.setZero();
      exponent += getExpLaplaceGradIJ(i, j, gi, gj,
                                      laplacian[i], laplacian[j],
                                      d.coord[i], d.coord[j],
                                      params, gradHelper, 1.0, true);
      Gradient(i,0)+= gi[0]; Gradient(i,1)+= gi[1]; Gradient(i,2) += gi[2];
      Gradient(j,0)+= gj[0]; Gradient(j,1)+= gj[1]; Gradient(j,2) += gj[2];
      //cout << i<<"  "<<j<<"  "<<exponent <<endl;
    }

  //cout << exponent <<endl;
  //exit(0);
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



void GeneralJastrow::UpdateLaplaceGrad(MatrixXd& Gradient,
                                       VectorXd& laplacian,
                                       const rDeterminant& d,
                                       const Vector3d& oldCoord,
                                       int i,
                                       const double * params,
                                       double * gradHelper) const {
  
  Vector3d gi, gj;
  double laplacei=0., laplacej = 0. ;

  for (int j=0; j<d.nelec; j++) {
    if (j == i) continue;

    laplacei = 0; laplacej = 0;
    gi.setZero(); gj.setZero();
    getExpLaplaceGradIJ(i, j, gi, gj,
                        laplacei, laplacej,
                        d.coord[i], d.coord[j],
                        params, gradHelper, 1.0, true);
    
    Gradient(i,0) += gi[0];  Gradient(i,1) += gi[1];  Gradient(i,2) += gi[2];
    Gradient(j,0) += gj[0];  Gradient(j,1) += gj[1];  Gradient(j,2) += gj[2];
    laplacian[i] += laplacei;  laplacian[j] += laplacej;

    laplacei = 0; laplacej = 0;
    gi.setZero(); gj.setZero();
    getExpLaplaceGradIJ(i, j, gi, gj,
                        laplacei, laplacej,
                        oldCoord, d.coord[j],
                        params, gradHelper, -1.0, true);
    
    Gradient(i,0) -= gi[0];  Gradient(i,1) -= gi[1];  Gradient(i,2) -= gi[2];
    Gradient(j,0) -= gj[0];  Gradient(j,1) -= gj[1];  Gradient(j,2) -= gj[2];
    laplacian[i] -= laplacei;  laplacian[j] -= laplacej;
    
  }
}


void GeneralJastrow::OverlapWithGradient(VectorXd& grad, int& index,
                                         const vector<double>& gradHelper) const {
  
  for (int i=0; i<gradHelper.size(); i++) {
    //cout << i<<"  "<<gradHelper[i]<<"  "<<index<<endl;
    if (fixed[i] == 0) {
      grad[index] = gradHelper[i]; index++;
    }
    else {
      grad[index] = 0; index++;
    }
  }
  //exit(0);
}













double getAlpha(int i, int j) {
  if (i % rDeterminant::nalpha == j % rDeterminant::nalpha) return 0.25;
  else return 0.5;
}
void TwoBodyLaplaceHelper(const double& rij, double& alpha, const double * params,
                          const double& beta, int minOrder, int maxOrder,
                          double& laplace) {

  double rijbar    = rij/(1. + beta*rij);
  double DrbarDr   = 1./pow(1. + beta*rij, 2);
  double D2rbarDr2 = -2*beta/pow(1. + beta*rij, 3);
  
  double DexpDrbar   = alpha/pow(1 + params[0]*rijbar, 2);
  double D2expDrbar2 = -2*alpha*params[0]/pow(1. + params[0]* rijbar, 3);
  
  for (int o = minOrder; o < maxOrder+1; o++) {
    DexpDrbar   += o * params[o-1] * pow(rijbar, o-1);
    D2expDrbar2 += o * (o-1) * params[o-1] * pow(rijbar, o-2);
  }
  
  double DexpDr   = DexpDrbar * DrbarDr;
  double D2expDr2 = D2expDrbar2 * pow(DrbarDr, 2) + DexpDrbar * D2rbarDr2;
      
  laplace = D2expDr2 + DexpDr * 2/rij ;  
}

void TwoBodyGradHelper(const double& rij, double& alpha, const double * params,
                       const double& beta, int minOrder, int maxOrder,
                       double& xij, double& yij, double& zij,
                       double& gx, double& gy, double& gz) {

  double rijbar = rij/(1. + beta*rij);
  double DrbarDr = 1./pow(1. + beta*rij, 2);
  
  double DexpDrbar = alpha/pow(1 + params[0]*rijbar, 2);
  for (int o = minOrder; o < maxOrder+1; o++) 
    DexpDrbar += o * params[o-1] * pow(rijbar, o-1);
  double DexpDr = DexpDrbar * DrbarDr; 
  
  
  gx = DexpDr * xij/rij;
  gy = DexpDr * yij/rij;
  gz = DexpDr * zij/rij;

}


EEJastrow::EEJastrow() {
  beta = 1.0;
  //beta = 0.5;
}

double EEJastrow::exponential(const MatrixXd& Rij, const MatrixXd& RiN,
                              int maxOrder, const double * params,
                              double * gradHelper) const {

  double exponent = 0.0;
  for (int i=0; i<Rij.rows(); i++)
    for (int j=0; j<i; j++) {

      double rijbar = Rij(i,j) /(1. + beta*Rij(i,j) );

      double alpha = getAlpha(i, j);
      exponent += alpha * rijbar/ (1. + params[0]* rijbar);

      gradHelper[0] += -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      //gradHelper[0] += alpha*rijbar/pow(1+params[0]*rijbar,1);
      //sum over higher orders
      for (int o = 2; o < maxOrder+1; o++) {
        gradHelper[o-1] += pow(rijbar, o);
        exponent += params[o-1] * pow(rijbar, o);;
      }      
    }

  return exponent;  
}

double EEJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d, int maxOrder,
                               const double * params, double * gradHelper) const {
  
  double diff = 0.0;
  for (int j=0; j<d.nelec; j++) {
    if (i == j) continue;
    double rijold = pow( pow(d.coord[i][0] - d.coord[j][0], 2) +
                      pow(d.coord[i][1] - d.coord[j][1], 2) +
                      pow(d.coord[i][2] - d.coord[j][2], 2), 0.5);
    
    double rij = pow( pow(newcoord[0] - d.coord[j][0], 2) +
                          pow(newcoord[1] - d.coord[j][1], 2) +
                          pow(newcoord[2] - d.coord[j][2], 2), 0.5);

    double alpha = getAlpha(i, j);

    double rijbar = rij/(1.+ beta*rij),
        rijbarold = rijold/(1. + beta*rijold);

    diff += alpha * (rijbar/(1.+params[0]*rijbar) -
                     rijbarold/(1.+params[0]*rijbarold));

    //sum over higher orders
    for (int o = 2; o < maxOrder+1; o++) {
      diff += params[o-1] * (pow(rijbar, o) - pow(rijbarold, o));
    }      
  }

  return diff;
}

void EEJastrow::InitGradient(MatrixXd& Gradient,
                             const MatrixXd& Rij,
                             const MatrixXd& RiN,
                             const rDeterminant& d,
                             int maxOrder, const double * params) const {

  double gx, gy, gz;
  int minOrder=2;
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<i; j++) {

      double alpha = getAlpha(i, j);

      double xij = d.coord[i][0] - d.coord[j][0],
             yij = d.coord[i][1] - d.coord[j][1],
             zij = d.coord[i][2] - d.coord[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, params, beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) += gx;  Gradient(i,1) += gy;  Gradient(i,2) += gz;
      
      Gradient(j,0) -= gx;  Gradient(j,1) -= gy;  Gradient(j,2) -= gz;
    }
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::InitLaplacian(VectorXd &laplacian,
                              const MatrixXd& Rij,
                              const MatrixXd& RiN,
                              const rDeterminant& d,
                              int maxOrder, const double * params) const {

  double laplace;  
  int minOrder = 2;
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<i; j++) {
      
      double alpha = getAlpha(i, j);

      TwoBodyLaplaceHelper(Rij(i,j), alpha, params, beta, minOrder, maxOrder, laplace);
      laplacian[i] += laplace;  laplacian[j] += laplace;      
    }    
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::UpdateGradient(MatrixXd& Gradient,
                               const MatrixXd& Rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord,
                               int i,
                               int maxOrder, const double * params,
                               double * gradHelper) const {
  
  double gx, gy, gz;
  int minOrder = 2;
  for (int j=0; j<d.nelec; j++) {
    if (j == i) continue;

    double alpha = getAlpha(i, j);

    //calculate the old contribution
    {
      double xij = oldCoord[0] - d.coord[j][0],
             yij = oldCoord[1] - d.coord[j][1],
             zij = oldCoord[2] - d.coord[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, params, beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) -= gx;  Gradient(i,1) -= gy;  Gradient(i,2) -= gz;
      
      Gradient(j,0) += gx;  Gradient(j,1) += gy;  Gradient(j,2) += gz;

      double rijbar = rij/(1.+beta*rij);
      //gradHelper[0] -= -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      gradHelper[0] -=  -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      for (int o = 2; o < maxOrder+1; o++) 
        gradHelper[o-1] -= pow(rijbar, o);
    }

    //calculate the new contribution
    {
      double xij = d.coord[i][0] - d.coord[j][0],
             yij = d.coord[i][1] - d.coord[j][1],
             zij = d.coord[i][2] - d.coord[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, params, beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) += gx;  Gradient(i,1) += gy;  Gradient(i,2) += gz;
      
      Gradient(j,0) -= gx;  Gradient(j,1) -= gy;  Gradient(j,2) -= gz;

      double rijbar = rij/(1.+beta*rij);
      //gradHelper[0] += -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      gradHelper[0] +=  -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      for (int o = 2; o < maxOrder+1; o++) 
        gradHelper[o-1] += pow(rijbar, o);
    }      
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::UpdateLaplacian(VectorXd &laplacian,
                                const MatrixXd& Rij,
                                const MatrixXd& RiN,
                                const rDeterminant& d,
                                const Vector3d& oldCoord,
                                int i,
                                int maxOrder, const double * params) const {

  double laplace ;
  int minOrder = 2;
  for (int j=0; j<d.nelec; j++) {
    if (j == i) continue;
    
    double alpha = getAlpha(i, j);
    //calculate the old contribution
    {

      double rij = pow( pow(oldCoord[0] - d.coord[j][0], 2) +
                        pow(oldCoord[1] - d.coord[j][1], 2) +
                        pow(oldCoord[2] - d.coord[j][2], 2), 0.5);

      TwoBodyLaplaceHelper(rij, alpha, params, beta, minOrder, maxOrder, laplace);
      laplacian[i] -= laplace;  laplacian[j] -= laplace;
    }

    //calculate the new contribution
    {
      TwoBodyLaplaceHelper(Rij(i,j), alpha, params, beta, minOrder, maxOrder, laplace);
      laplacian[i] += laplace;  laplacian[j] += laplace;
    }
  }    
}

void EEJastrow::OverlapWithGradient(VectorXd& grad, int& index,
                                    int minOrder, int maxOrder,
                                    const double * gradHelper) const {
  /*
  for (int i=0; i<maxOrder; i++)
    gradHelper[i] = 0.0;

  for (int i=0; i<Rij.rows(); i++)
    for (int j=0; j<i; j++) {

      double rijbar = Rij(i,j) /(1. + beta*Rij(i,j) );

      double alpha = getAlpha(i, j);

      gradHelper[0] -= alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      //sum over higher orders
      for (int o = 2; o < maxOrder+1; o++) {
        gradHelper[o-1] += pow(rijbar, o);
      }      
    }

  grad[index] = gradHelper[0]; index++;
  for (int o = minOrder; o < maxOrder+1; o++) {
    grad[index] = gradHelper[o-1] ;
    index++;
  }
  */

  grad[index] = gradHelper[0]; index++;
  for (int o = minOrder; o < maxOrder+1; o++) {
    grad[index] = gradHelper[o-1] ;
    index++;
  }

}


ENJastrow::ENJastrow() {
  beta = 1.0;
  //beta = 0.5;
}

double ENJastrow::exponential(const MatrixXd& Rij, const MatrixXd& RiN,
                              int maxOrder, const double * params,
                              double * gradHelper) const {

  double exponent = 0.0;

  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;
  
  for (int i=0; i<RiN.rows(); i++)
    for (int j=0; j<RiN.cols(); j++) {
      
      double rijbar = RiN(i,j) /(1. + beta*RiN(i,j) );

      gradHelper[nTerm*j + 0] = 0.0; 
      //sum over higher orders
      for (int o = 2; o < maxOrder+1; o++) {
        gradHelper[nTerm*j + o-1] += pow(rijbar, o);
        exponent += params[nTerm*j + o-1] * pow(rijbar, o);;
      }      
    }

  return exponent;  
}

double ENJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d, int maxOrder,
                               const double * params, double * gradHelper) const {
  
  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;

  double diff = 0.0;
  for (int j=0; j<Ncoords.size(); j++) {

    double rijold = pow( pow(d.coord[i][0] - Ncoords[j][0], 2) +
                         pow(d.coord[i][1] - Ncoords[j][1], 2) +
                         pow(d.coord[i][2] - Ncoords[j][2], 2), 0.5);
    
    double rij = pow( pow(newcoord[0] - Ncoords[j][0], 2) +
                      pow(newcoord[1] - Ncoords[j][1], 2) +
                      pow(newcoord[2] - Ncoords[j][2], 2), 0.5);


    double rijbar = rij/(1.+ beta*rij),
        rijbarold = rijold/(1. + beta*rijold);

    //sum over higher orders
    for (int o = 2; o < maxOrder+1; o++) {
      diff += params[nTerm*j + o-1] * (pow(rijbar, o) - pow(rijbarold, o));
    }      
  }

  return diff;
}

void ENJastrow::InitGradient(MatrixXd& Gradient,
                             const MatrixXd& Rij,
                             const MatrixXd& RiN,
                             const rDeterminant& d,
                             int maxOrder, const double * params) const {

  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;
  
  double gx, gy, gz;
  int minOrder=2;
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<Ncoords.size(); j++) {

      double alpha = 0.0; //no first order term
      
      double xij = d.coord[i][0] - Ncoords[j][0],
             yij = d.coord[i][1] - Ncoords[j][1],
             zij = d.coord[i][2] - Ncoords[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, &params[nTerm*j],
                        beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) += gx;  Gradient(i,1) += gy;  Gradient(i,2) += gz;
    }
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::InitLaplacian(VectorXd &laplacian,
                              const MatrixXd& Rij,
                              const MatrixXd& RiN,
                              const rDeterminant& d,
                              int maxOrder, const double * params) const {

  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;

  double laplace;  
  int minOrder = 2;
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<Ncoords.size(); j++) {
      
      double alpha = 0.;//no first order term

      TwoBodyLaplaceHelper(RiN(i,j), alpha, &params[nTerm*j],
                           beta, minOrder, maxOrder, laplace);
      laplacian[i] += laplace;
    }    
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::UpdateGradient(MatrixXd& Gradient,
                               const MatrixXd& Rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord,
                               int i,
                               int maxOrder, const double * params,
                               double * gradHelper) const {
  
  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;

  double gx, gy, gz;
  int minOrder = 2;
  for (int j=0; j<Ncoords.size(); j++) {

    double alpha = 0.;

    //calculate the old contribution
    {
      double xij = oldCoord[0] - Ncoords[j][0],
             yij = oldCoord[1] - Ncoords[j][1],
             zij = oldCoord[2] - Ncoords[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, params+nTerm*j,
                        beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) -= gx;  Gradient(i,1) -= gy;  Gradient(i,2) -= gz;

      for (int o = 2; o < maxOrder+1; o++) 
        gradHelper[nTerm*j+o-1] -= pow(rij/(1.+beta*rij), o);
    }

    //calculate the new contribution
    {
      double xij = d.coord[i][0] - Ncoords[j][0],
             yij = d.coord[i][1] - Ncoords[j][1],
             zij = d.coord[i][2] - Ncoords[j][2];
      double rij = pow( xij*xij + yij*yij + zij*zij, 0.5);

      TwoBodyGradHelper(rij, alpha, params+nTerm*j,
                        beta, minOrder, maxOrder,
                        xij, yij, zij,
                        gx, gy, gz);

      Gradient(i,0) += gx;  Gradient(i,1) += gy;  Gradient(i,2) += gz;

      for (int o = 2; o < maxOrder+1; o++) 
        gradHelper[nTerm*j+o-1] += pow(rij/(1.+beta*rij), o);
    }      
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::UpdateLaplacian(VectorXd &laplacian,
                                const MatrixXd& Rij,
                                const MatrixXd& RiN,
                                const rDeterminant& d,
                                const Vector3d& oldCoord,
                                int i,
                                int maxOrder, const double * params) const {

  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;

  double laplace ;
  int minOrder = 2;
  for (int j=0; j<Ncoords.size(); j++) {

    double alpha = 0.0;
    //calculate the old contribution
    {

      double rij = pow( pow(oldCoord[0] - Ncoords[j][0], 2) +
                        pow(oldCoord[1] - Ncoords[j][1], 2) +
                        pow(oldCoord[2] - Ncoords[j][2], 2), 0.5);

      TwoBodyLaplaceHelper(rij, alpha, params+nTerm*j,
                           beta, minOrder, maxOrder, laplace);
      laplacian[i] -= laplace;
    }

    //calculate the new contribution
    {
      TwoBodyLaplaceHelper(RiN(i,j), alpha, params+nTerm+j,
                           beta, minOrder, maxOrder, laplace);
      laplacian[i] += laplace;
    }
  }    
}

void ENJastrow::OverlapWithGradient(VectorXd& grad, int& index,
                                    int minOrder, int maxOrder,
                                    const double * gradHelper) const {

  int nTerm = maxOrder - 1;
  for (int j=0; j<Ncoords.size(); j++) {
    grad[index] = 0; index++;
    for (int o = minOrder; o < maxOrder+1; o++) {
      grad[index] = gradHelper[j*nTerm+o-1] ; index++;
    }
  }
}

/*
ENJastrow::ENJastrow()
{
  Ncoords = schd.Ncoords;
  Ncharge = schd.Ncharge;
  //alpha.resize(schd.Ncoords.size(), 500.);
  //alpha.resize(schd.Ncoords.size(), 100.);
  alpha.resize(schd.Ncoords.size(), 1.);
}

double ENJastrow::exponential(const MatrixXd& rij, const MatrixXd& RiN) const {
  double exponent = 0.0;
  for (int i=0; i<RiN.rows(); i++)
    for (int j=0; j<RiN.cols(); j++) {
      double rijbar = RiN(i,j) /(1. + alpha[j] * RiN(i,j) );
      exponent += -Ncharge[j]* rijbar;
    }
  return exponent;  
}

double ENJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d) {
  double diff = 0.0;
  for (int j=0; j<Ncoords.size(); j++) {
    double rij = pow( pow(d.coord[i][0] - Ncoords[j][0], 2) +
                      pow(d.coord[i][1] - Ncoords[j][1], 2) +
                      pow(d.coord[i][2] - Ncoords[j][2], 2), 0.5);
    
    double rijcopy = pow( pow(newcoord[0] - Ncoords[j][0], 2) +
                          pow(newcoord[1] - Ncoords[j][1], 2) +
                          pow(newcoord[2] - Ncoords[j][2], 2), 0.5);
    
    diff += -Ncharge[j] * (rijcopy/(1.+alpha[j]*rijcopy) - rij/(1.+alpha[j]*rij));
  }
  return diff;
}

void ENJastrow::InitGradient(MatrixXd& Gradient,
                             const MatrixXd& rij,
                             const MatrixXd& RiN, const rDeterminant& d) {
  
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<Ncoords.size(); j++) {

      double gx = -Ncharge[j]*(d.coord[i][0] - Ncoords[j][0])/RiN(i,j)/pow( 1+ alpha[j]*RiN(i, j),2);
      double gy = -Ncharge[j]*(d.coord[i][1] - Ncoords[j][1])/RiN(i,j)/pow( 1+ alpha[j]*RiN(i, j),2);
      double gz = -Ncharge[j]*(d.coord[i][2] - Ncoords[j][2])/RiN(i,j)/pow( 1+ alpha[j]*RiN(i, j),2);
      Gradient(i,0) += gx;
      Gradient(i,1) += gy;
      Gradient(i,2) += gz;
    }
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::InitLaplacian(VectorXd &laplacian,
                              const MatrixXd& rij,
                              const MatrixXd& RiN,
                              const rDeterminant& d) {
  
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<Ncoords.size(); j++) {
      
      //calculate the contribution
      double newTerm = + 2*alpha[j]*Ncharge[j]/(pow(1+alpha[j]*RiN(i,j), 3))
          - 2*Ncharge[j]/RiN(i,j)/(pow(1+alpha[j]*RiN(i,j), 2));
      
      //add the contribution
      laplacian[i] += newTerm;
    }    
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::UpdateGradient(MatrixXd& Gradient,
                               const MatrixXd& rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord,
                               int elecI) {
  
  for (int j=0; j<Ncoords.size(); j++) {
    //calculate the old contribution
    double oldriN = pow( pow(oldCoord[0] - Ncoords[j][0], 2) +
                         pow(oldCoord[1] - Ncoords[j][1], 2) +
                         pow(oldCoord[2] - Ncoords[j][2], 2), 0.5);
    double gradxij = -Ncharge[j]*(oldCoord[0] - Ncoords[j][0])/oldriN/pow( 1+ alpha[j]*oldriN,2);
    double gradyij = -Ncharge[j]*(oldCoord[1] - Ncoords[j][1])/oldriN/pow( 1+ alpha[j]*oldriN,2);
    double gradzij = -Ncharge[j]*(oldCoord[2] - Ncoords[j][2])/oldriN/pow( 1+ alpha[j]*oldriN,2);

    //Remove the old contribution
    Gradient(elecI,0) -= gradxij;
    Gradient(elecI,1) -= gradyij;
    Gradient(elecI,2) -= gradzij;
    
    //calculate the new contribution
    gradxij = -Ncharge[j]*(d.coord[elecI][0] - Ncoords[j][0])/RiN(elecI,j)/pow( 1+ alpha[j]*RiN(elecI, j),2);
    gradyij = -Ncharge[j]*(d.coord[elecI][1] - Ncoords[j][1])/RiN(elecI,j)/pow( 1+ alpha[j]*RiN(elecI, j),2);
    gradzij = -Ncharge[j]*(d.coord[elecI][2] - Ncoords[j][2])/RiN(elecI,j)/pow( 1+ alpha[j]*RiN(elecI, j),2);
    
    //Add the new contribution
    Gradient(elecI,0) += gradxij;
    Gradient(elecI,1) += gradyij;
    Gradient(elecI,2) += gradzij;
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void ENJastrow::UpdateLaplacian(VectorXd &laplacian,
                                const MatrixXd& rij,
                                const MatrixXd& RiN,
                                const rDeterminant& d,
                                const Vector3d& oldCoord,
                                int elecI) {
  
  double term2 = 0.0;
  for (int j=0; j<Ncoords.size(); j++) {
    
    //calculate the old contribution
    double oldriN = pow( pow(oldCoord[0] - Ncoords[j][0], 2) +
                         pow(oldCoord[1] - Ncoords[j][1], 2) +
                         pow(oldCoord[2] - Ncoords[j][2], 2), 0.5);
    double oldTerm = + 2*alpha[j]*Ncharge[j]/(pow(1+alpha[j]*oldriN, 3))
        - 2*Ncharge[j]/oldriN/(pow(1+alpha[j]*oldriN, 2));
    
    //remove the old contribution
    laplacian[elecI] -= oldTerm;

    //calculate the new contribution
    double newTerm = + 2*alpha[j]*Ncharge[j]/(pow(1+alpha[j]*RiN(elecI,j), 3))
        - 2*Ncharge[j]/RiN(elecI,j)/(pow(1+alpha[j]*RiN(elecI,j), 2));

    //add the old contribution
    laplacian[elecI] += newTerm;
    
  }    
}

long ENJastrow::getNumVariables() {return 1;}

void ENJastrow::getVariables(Eigen::VectorXd& v, int& numVars)  {
  for (int i=0; i<Ncoords.size(); i++)
    v[numVars+i] = alpha[i];
  numVars += Ncoords.size();
}

void ENJastrow::updateVariables(const Eigen::VectorXd& v, int &numVars) {
  for (int i=0; i<Ncoords.size(); i++)
    alpha[i] = v[numVars+i] ;
  numVars += Ncoords.size();
}

void ENJastrow::OverlapWithGradient(const MatrixXd& rij, const MatrixXd& RiN,
                                    const rDeterminant& d, VectorXd& grad,
                                    const double& ovlp, int& index) {
  
  for (int j=0; j<Ncoords.size(); j++) {      
    for (int i=0; i<RiN.rows(); i++)
      grad[index] += Ncharge[j] * pow(RiN(i,j), 2) /pow(1 + alpha[j]*RiN(i, j), 2);
    
    index++;
  }
}

void ENJastrow::printVariables() {
  cout << "ENJastrow"<<endl;
  for (int i=0; i<Ncoords.size(); i++)
    cout << alpha[i]<<"  ";
  cout << endl;
}
*/
