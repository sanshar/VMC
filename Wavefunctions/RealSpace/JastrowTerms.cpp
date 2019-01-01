#include "JastrowTerms.h"

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
  
  double DexpDrbar   = alpha;///(1 + params[0]*rijbar);
  double D2expDrbar2 = 0.0;
  
  for (int o = minOrder; o < maxOrder+1; o++) {
    DexpDrbar   += o * params[o-2] * pow(rijbar, o-1);
    D2expDrbar2 += o * (o-1) * params[o-2] * pow(rijbar, o-2);
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
  
  double DexpDrbar = alpha;///(1 + params[0]*rijbar);
  for (int o = minOrder; o < maxOrder+1; o++) 
    DexpDrbar += o * params[o-2] * pow(rijbar, o-1);
  double DexpDr = DexpDrbar * DrbarDr; 
  
  
  gx = DexpDr * xij/rij;
  gy = DexpDr * yij/rij;
  gz = DexpDr * zij/rij;

}

EEJastrow::EEJastrow() {
  beta = 1.0;
}

double EEJastrow::exponential(const MatrixXd& Rij, const MatrixXd& RiN,
                              int maxOrder, const double * params,
                              double * values) const {

  double exponent = 0.0;
  for (int i=0; i<Rij.rows(); i++)
    for (int j=0; j<i; j++) {

      double rijbar = Rij(i,j) /(1. + beta*Rij(i,j) );

      double alpha = getAlpha(i, j);
      exponent += alpha * rijbar; // (1. + params[0]* rijbar);

      //values[0] += -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      //sum over higher orders
      for (int o = 2; o < maxOrder+1; o++) {
        values[o-2] += pow(rijbar, o);
        exponent += params[o-2] * pow(rijbar, o);;
      }      
    }

  return exponent;  
}

double EEJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d, int maxOrder,
                               const double * params, double * values) const {
  
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

    diff += alpha * (rijbar - rijbarold);

    //sum over higher orders
    for (int o = 2; o < maxOrder+1; o++) {
      diff += params[o-2] * (pow(rijbar, o) - pow(rijbarold, o));
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
                               double * values) const {
  
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
      for (int o = 2; o < maxOrder+1; o++) 
        values[o-2] -= pow(rijbar, o);
      //for (int o = 2; o < maxOrder+1; o++) 
      //values[o-2] -= pow(rij/(1.+beta*rij), o);
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

      //double rijbar = rij/(1.+beta*rij);
      //values[0] += -alpha*rijbar*rijbar/pow(1+params[0]*rijbar,2);
      double rijbar = rij/(1.+beta*rij);
      for (int o = 2; o < maxOrder+1; o++) 
        values[o-2] += pow(rijbar, o);
      //for (int o = 2; o < maxOrder+1; o++) 
      //values[o-2] += pow(rij/(1.+beta*rij), o);
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
                                    const double * values) const {

  //grad[index] = values[0]; index++;
  for (int o = minOrder; o < maxOrder+1; o++) {
    grad[index] = values[o-2] ;
    index++;
  }
}


ENJastrow::ENJastrow() {
  beta = 1.0;
}

double ENJastrow::exponential(const MatrixXd& Rij, const MatrixXd& RiN,
                              int maxOrder, const double * params,
                              double * values) const {

  double exponent = 0.0;

  int nN = schd.Ncharge.size();
  int nTerm = maxOrder - 1;
  
  for (int i=0; i<RiN.rows(); i++)
    for (int j=0; j<RiN.cols(); j++) {
      
      double rijbar = RiN(i,j) /(1. + beta*RiN(i,j) );

      //sum over higher orders
      for (int o = 2; o < maxOrder+1; o++) {
        values[nTerm*j + o-2] += pow(rijbar, o);
        exponent += params[nTerm*j + o-2] * pow(rijbar, o);;
      }      
    }

  return exponent;  
}

double ENJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d, int maxOrder,
                               const double * params, double * values) const {
  
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
      diff += params[nTerm*j + o-2] * (pow(rijbar, o) - pow(rijbarold, o));
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
                               double * values) const {
  
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
        values[nTerm*j+o-2] -= pow(rij/(1.+beta*rij), o);
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
        values[nTerm*j+o-2] += pow(rij/(1.+beta*rij), o);
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
                                    const double * values) const {

  int nTerm = maxOrder - 1;
  for (int j=0; j<Ncoords.size(); j++)
    for (int o = minOrder; o < maxOrder+1; o++) {
      grad[index] = values[j*nTerm+o-2] ;
      index++;
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
