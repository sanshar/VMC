#include "JastrowTerms.h"

double EEJastrow::exponential(const MatrixXd& rij, const MatrixXd& RiN) const {
  double exponent = 0.0;
  for (int i=0; i<rij.rows(); i++)
    for (int j=0; j<i; j++) {
      double rijbar = rij(i,j) /(1. + beta*rij(i,j) );
      exponent += alpha* rijbar;
    }
  return exponent;  
}

double EEJastrow::exponentDiff(int i, const Vector3d &newcoord,
                               const rDeterminant &d) {
  double diff = 0.0;
  for (int j=0; j<d.nelec; j++) {
    double rij = pow( pow(d.coord[i][0] - d.coord[j][0], 2) +
                      pow(d.coord[i][1] - d.coord[j][1], 2) +
                      pow(d.coord[i][2] - d.coord[j][2], 2), 0.5);
    
    double rijcopy = pow( pow(newcoord[0] - d.coord[j][0], 2) +
                          pow(newcoord[1] - d.coord[j][1], 2) +
                          pow(newcoord[2] - d.coord[j][2], 2), 0.5);
    
    diff += alpha * (rijcopy/(1.+ beta*rijcopy) - rij/(1.+beta*rij));
  }
  return diff;
}

void EEJastrow::InitGradient(MatrixXd& Gradient,
                             const MatrixXd& rij,
                             const MatrixXd& RiN, const rDeterminant& d) {
  
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<i; j++) {
      if (j == i) continue;
      double gx = alpha*(d.coord[i][0] - d.coord[j][0])/rij(i, j)/pow( 1+ beta*rij(i,j),2);
      double gy = alpha*(d.coord[i][1] - d.coord[j][1])/rij(i, j)/pow( 1+ beta*rij(i,j),2);
      double gz = alpha*(d.coord[i][2] - d.coord[j][2])/rij(i, j)/pow( 1+ beta*rij(i,j),2);
      Gradient(i,0) += gx;
      Gradient(i,1) += gy;
      Gradient(i,2) += gz;
      
      Gradient(j,0) -= gx;
      Gradient(j,1) -= gy;
      Gradient(j,2) -= gz;
    }
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::InitLaplacian(VectorXd &laplacian,
                              const MatrixXd& rij,
                              const MatrixXd& RiN,
                              const rDeterminant& d) {
  
  for (int i=0; i<d.nelec; i++) {
    for (int j=0; j<i; j++) {
      
      //calculate the new contribution
      double newTerm =  - 2*beta*alpha/(pow(1+beta*rij(i, j), 3))
          + 2*alpha/rij(i, j)/(pow(1+beta*rij(i, j), 2));
      
      //add the old contribution
      laplacian[i] += newTerm;
      laplacian[j] += newTerm;
      
    }    
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::UpdateGradient(MatrixXd& Gradient,
                               const MatrixXd& rij,
                               const MatrixXd& RiN,
                               const rDeterminant& d,
                               const Vector3d& oldCoord,
                               int elecI) {
  
  for (int j=0; j<d.nelec; j++) {
    if (j == elecI) continue;
    //calculate the old contribution
    double oldrij = pow( pow(oldCoord[0] - d.coord[j][0], 2) +
                         pow(oldCoord[1] - d.coord[j][1], 2) +
                         pow(oldCoord[2] - d.coord[j][2], 2), 0.5);

    double gradxij = alpha*(oldCoord[0] - d.coord[j][0])/oldrij/pow( 1+ beta*oldrij,2);
    double gradyij = alpha*(oldCoord[1] - d.coord[j][1])/oldrij/pow( 1+ beta*oldrij,2);
    double gradzij = alpha*(oldCoord[2] - d.coord[j][2])/oldrij/pow( 1+ beta*oldrij,2);

    //Remove the old contribution
    Gradient(elecI,0) -= gradxij;
    Gradient(elecI,1) -= gradyij;
    Gradient(elecI,2) -= gradzij;
    
    Gradient(j,0) += gradxij;
    Gradient(j,1) += gradyij;
    Gradient(j,2) += gradzij;
    
    //calculate the new contribution
    gradxij = alpha*(d.coord[elecI][0] - d.coord[j][0])/rij(elecI, j)/pow( 1+ beta*rij(elecI,j),2);
    gradyij = alpha*(d.coord[elecI][1] - d.coord[j][1])/rij(elecI, j)/pow( 1+ beta*rij(elecI,j),2);
    gradzij = alpha*(d.coord[elecI][2] - d.coord[j][2])/rij(elecI, j)/pow( 1+ beta*rij(elecI,j),2);
    
    //Add the new contribution
    Gradient(elecI,0) += gradxij;
    Gradient(elecI,1) += gradyij;
    Gradient(elecI,2) += gradzij;
    
    Gradient(j,0) -= gradxij;
    Gradient(j,1) -= gradyij;
    Gradient(j,2) -= gradzij;
  }
}

//J = exp( sum_ij uij)
//\sum_j Nabla^2_i uij  
void EEJastrow::UpdateLaplacian(VectorXd &laplacian,
                                const MatrixXd& rij,
                                const MatrixXd& RiN,
                                const rDeterminant& d,
                                const Vector3d& oldCoord,
                                int elecI) {
  
  for (int j=0; j<d.nelec; j++) {
    if (j == elecI) continue;
    
    //calculate the old contribution
    double oldrij = pow( pow(oldCoord[0] - d.coord[j][0], 2) +
                         pow(oldCoord[1] - d.coord[j][1], 2) +
                         pow(oldCoord[2] - d.coord[j][2], 2), 0.5);
    double oldTerm = - 2*beta*alpha/(pow(1+beta*oldrij, 3))
        + 2*alpha/oldrij/(pow(1+beta*oldrij, 2));
    
    //remove the old contribution
    laplacian[elecI] -= oldTerm;
    laplacian[j] -= oldTerm;

    //cout << laplacian[elecI]<<"  "<<laplacian[j]<<endl;
    //calculate the new contribution
    double newTerm =  - 2*beta*alpha/(pow(1+beta*rij(elecI, j), 3))
        + 2*alpha/rij(elecI, j)/(pow(1+beta*rij(elecI, j), 2));
    
    //add the old contribution
    laplacian[elecI] += newTerm;
    laplacian[j] += newTerm;
    
  }    
}

long EEJastrow::getNumVariables() {return 1;}

void EEJastrow::getVariables(Eigen::VectorXd& v, int& numVars)  {
  v[numVars] = beta;
  numVars += 1;
}

void EEJastrow::updateVariables(const Eigen::VectorXd& v, int &numVars) {
  beta = v[numVars];
  numVars += 1;
}

void EEJastrow::OverlapWithGradient(const MatrixXd& rij, const MatrixXd& RiN,
                                    const rDeterminant& d, VectorXd& grad,
                                    const double& ovlp, int& index) {
  
  for (int j=0; j<rij.rows(); j++) 
    for (int i=0; i<j; i++)
      grad[index] -= alpha * pow(rij(i,j), 2) /pow(1 + beta*rij(i, j), 2);
  
  index++;
}

void EEJastrow::printVariables() {
  cout << "EEJastrow"<<endl;
  cout << beta<<endl;
}

ENJastrow::ENJastrow()
{
  Ncoords = schd.Ncoords;
  Ncharge = schd.Ncharge;
  alpha.resize(schd.Ncoords.size(), 1000.);
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
    gradxij = -Ncharge[j]*(d.coord[elecI][1] - Ncoords[j][1])/RiN(elecI,j)/pow( 1+ alpha[j]*RiN(elecI, j),2);
    gradxij = -Ncharge[j]*(d.coord[elecI][2] - Ncoords[j][2])/RiN(elecI,j)/pow( 1+ alpha[j]*RiN(elecI, j),2);
    
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
      grad[index] += 0.0*Ncharge[j] * pow(RiN(i,j), 2) /pow(1 + alpha[j]*RiN(i, j), 2);
    
    index++;
  }
}

void ENJastrow::printVariables() {
  cout << "ENJastrow"<<endl;
  for (int i=0; i<Ncoords.size(); i++)
    cout << alpha[i]<<"  ";
  cout << endl;
}
