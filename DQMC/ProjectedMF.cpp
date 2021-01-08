#include "ProjectedMF.h"
#include "Determinants.h"
#include "input.h"
#include "global.h"
#include "amsgrad.h"
#include "DQMCMatrixElements.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <iomanip>
using namespace std;
using namespace Eigen;

void applyProjectorSz(MatrixXcd& bra, vector<MatrixXcd>& ketvec, vector<complex<double>>& coeffs, double Sz, int ngrid) {
  
    int norbs = Determinant::norbs;
    double m = 1.*Sz/2.;
    int count = 0;
    complex<double> iImag(0, 1.0);

    coeffs.resize(2*ngrid);
    ketvec.resize(2*ngrid); 
    for (int i=0; i<ngrid; i++) {
        double gamma = 2. * M_PI * i / ngrid;

        complex<double> coeff = exp(-iImag * gamma * m)/(1. * ngrid * 2 * M_PI);
        coeffs[i] = coeff/sqrt(2.);    
        coeffs[ngrid+i] = coeff/sqrt(2.);//std::conj(coeff)/sqrt(2.);    

        VectorXcd phi = VectorXcd::Ones(2*norbs);

        phi.segment(0,norbs)     *= exp( iImag * gamma/2.);
        phi.segment(norbs,norbs) *= exp(-iImag * gamma/2.);
        ketvec[i] = phi.asDiagonal() * bra;
        ketvec[ngrid+i] = phi.asDiagonal() * bra.conjugate();
        //cout << ketvec[i]<<endl<<endl<<endl;
    }

}

void applyProjectorSsq(MatrixXcd& bra, vector<MatrixXcd>& ketvec, vector<complex<double>>& coeffs, double Sz, int ngrid) {
  
    int norbs = Determinant::norbs;
    double m = 1.*Sz/2.;
    int count = 0;
    complex<double> iImag(0, 1.0);

    coeffs.resize(2*ngrid*ngrid*ngrid);
    ketvec.resize(2*ngrid*ngrid*ngrid); 

    int index = 0;
    for (int i=0; i<ngrid; i++) {
        for (int j=0; j<ngrid; j++) {
            for (int k=0; k<ngrid; k++) {

                double alpha = 2. * M_PI * i / ngrid, 
                       beta =       M_PI * j / ngrid, 
                       gamma = 2. * M_PI * k / ngrid;

                complex<double> coeff = exp(-iImag * alpha * m)/(1. * ngrid * 2. * M_PI)  * exp(-iImag * gamma * m)/(1. * ngrid * 2 * M_PI);
                complex<double> wignerMat = 1.0/(ngrid * M_PI); //assume s= 0 m and k = 0

                coeffs[index] = coeff * wignerMat /sqrt(2.);    
                coeffs[ngrid*ngrid*ngrid+index] = coeff * wignerMat/sqrt(2.);   

                VectorXcd phik = VectorXcd::Ones(2*norbs);
                MatrixXcd phij = MatrixXcd::Zero(2*norbs, 2*norbs);
                VectorXcd phii = VectorXcd::Ones(2*norbs);

                //Sz
                phik.segment(0,norbs)     *= exp( iImag * gamma/2.);
                phik.segment(norbs,norbs) *= exp(-iImag * gamma/2.);

                //Sy
                phij.block(0,norbs,norbs,norbs) =  beta * MatrixXcd::Identity(norbs, norbs) / 2.;
                phij.block(norbs,0,norbs,norbs) = -beta * MatrixXcd::Identity(norbs, norbs) / 2.;
                phij = phij.exp();

                //Sz
                phii.segment(0,norbs)     *= exp( iImag * alpha/2.);
                phii.segment(norbs,norbs) *= exp(-iImag * alpha/2.);

                ketvec[index]       = phii.asDiagonal() * phij * phik.asDiagonal() * bra;
                ketvec[ngrid*ngrid*ngrid+index] = phii.asDiagonal() * phij * phik.asDiagonal() * bra.conjugate();
                index++;
            }
        }
    }

}

void copyToMat(VectorXd& vars, MatrixXcd& ref){
    int norbs = ref.rows(), nelec = ref.cols();
    int index = 0;
    for (int i=0; i<norbs; i++) {
      for (int j=0; j<nelec; j++) {
        ref(i,j) = complex<double>(vars[index], vars[norbs*nelec+index]);
        index++;
      }
    }
}

void copyFromMat(VectorXd& var, MatrixXcd& ref){
    int norbs = ref.rows(), nelec = ref.cols();
    int index = 0;
    for (int i=0; i<norbs; i++) {
      for (int j=0; j<nelec; j++) {
        var[index]             = ref(i,j).real();
        var[norbs*nelec+index] = ref(i,j).imag();
        index++;
      }
    }
}

complex<double> getEnergyProjected(MatrixXcd& ref, double enuc, MatrixXd& h1, vector<MatrixXd>& chol) {
    size_t norbs = Determinant::norbs;
    size_t nalpha = Determinant::nalpha;
    size_t nbeta = Determinant::nbeta;
    int ngrid = schd.ngrid;

    vector<complex<double>> coeffs; vector<MatrixXcd> ket;
    //applyProjectorSz(ref, ket, coeffs, 0, ngrid);
    applyProjectorSsq(ref, ket, coeffs, 0, ngrid);
    MatrixXcd refAd = ref.adjoint(), refT = ref.transpose();

    MatrixXcd  S;
    complex<double> O = 0, E=0.;
    
    for (int i=commrank; i<ket.size(); i+=commsize) {
      complex<double> Ei = calcHamiltonianElement(refAd, ket[i], enuc, h1, chol);
      S = refAd * ket[i];
      complex<double> Oi = S.determinant();
      E += coeffs[i] * (Ei * Oi).real();
      O += coeffs[i] * Oi.real();
    }

    MPI_Allreduce(MPI_IN_PLACE, &E, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &O, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    complex<double> E0 = E/O;
    return E0;
}

complex<double> getGradientProjected(MatrixXcd& ref, double enuc, MatrixXd& h1, vector<MatrixXd>& chol, MatrixXcd& Grad) {
    size_t norbs = Determinant::norbs;
    size_t nalpha = Determinant::nalpha;
    size_t nbeta = Determinant::nbeta;
    int ngrid = schd.ngrid;

    vector<complex<double>> coeffs; vector<MatrixXcd> ket;
    applyProjectorSsq(ref, ket, coeffs, 0, ngrid);
    MatrixXcd refAd = ref.adjoint(), refT = ref.transpose();

    MatrixXcd GradEi=0.*ref, GradOi=0.*ref, Gradnum=0.*ref, Gradden=0.*ref, S;
    complex<double> O = 0, E=0.;
    
    for (int i=commrank; i<ket.size(); i+=commsize) {
      complex<double> Ei = calcGradient(refAd, ket[i], enuc, h1, chol, GradEi);
      S = refAd * ket[i];
      complex<double> Oi = S.determinant();
      GradOi = Oi * (ket[i] * S.inverse());

      E += coeffs[i] * (Ei * Oi).real();
      O += coeffs[i] * Oi.real();

      Gradnum += coeffs[i] * (GradEi * Oi + Ei * GradOi);
      Gradden += coeffs[i] * GradOi;      
    }

    size_t mpisize = Gradnum.rows() * Gradnum.cols();
    MPI_Allreduce(MPI_IN_PLACE, &E, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &O, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Gradnum(0,0), mpisize, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Gradden(0,0), mpisize, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    complex<double> E0 = E/O;
    Grad = Gradnum/O - (E/O) * (Gradden/O);
    return E0;
}

void optimizeProjectedSlater(double enuc, MatrixXd& h1, vector<MatrixXd>& chol) {
    size_t norbs = Determinant::norbs;
    size_t nalpha = Determinant::nalpha;
    size_t nbeta = Determinant::nbeta;
    size_t nelec = nalpha + nbeta;
    cout << setprecision(10);

    MatrixXcd ref=MatrixXcd::Zero(2*norbs, nalpha+nbeta);

    MatrixXcd hf = MatrixXcd::Zero(2*norbs, 2*norbs);

    if (schd.hf == "uhf") {
        MatrixXcd hf = MatrixXcd::Zero(norbs, 2*norbs);
        readMat(hf, "uhf.txt");
        ref.block(0,0,norbs,nalpha) = 1.*hf.block(0,0,norbs, nalpha);// + 0.01*MatrixXcd::Random(2*norbs, nelec);
        ref.block(norbs,nalpha,norbs,nbeta) = 1.*hf.block(0,norbs,norbs, nbeta);// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    }
    else if (schd.hf == "ghf") {
        if (commrank == 0) cout << "read ghf"<<endl;
        readMat(hf, "ghf.txt");
        ref = 1.*hf.block(0,0,2*norbs, nelec);// + 0.01*MatrixXcd::Random(2*norbs, nelec);
    }

    //add noise    
    {
        MatrixXcd X = 0.01*MatrixXcd::Random(nelec, nelec);
        MatrixXcd U = (X - X.adjoint()).exp();
        ref = ref * U + 0.01*MatrixXcd::Random(2*norbs, nelec);
    }
    
    complex<double> initE = getEnergyProjected(ref, enuc, h1, chol);
    if (commrank == 0) cout <<"Initial Projected E "<< initE<<endl;
    

    //Lambda function for amsgrad
    auto LambdaGrad = [&chol, &h1, &enuc, norbs, nelec]
            (VectorXd& vars, VectorXd& grad, double& E0, 
            double& stddev, double& rt) {

        MatrixXcd bra(2*norbs, nelec), GradMat(2*norbs, nelec);
        copyToMat(vars, bra);
        E0 = getGradientProjected(bra, enuc, h1, chol, GradMat).real();
        copyFromMat(grad,GradMat);
    };

    VectorXd vars(4*norbs*nelec);
    copyFromMat(vars, ref);

    AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter, schd.avgIter);
    optimizer.optimize(vars, LambdaGrad, schd.restart);
    copyToMat(vars, ref);
    hf.block(0,0,2*norbs,nelec) = 1*ref;
    if (commrank == 0) writeMat(hf,"ghf.txt");
}
