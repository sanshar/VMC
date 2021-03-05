#include "input.h"
#include "RHF.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

RHF::RHF(Hamiltonian& ham, bool pleftQ, std::string fname) 
{
  MatrixXd hf = MatrixXd::Zero(ham.norbs, ham.norbs);
  readMat(hf, fname);
  det = hf.block(0, 0, ham.norbs, ham.nalpha);
  detT = det.adjoint();
  leftQ = pleftQ;
  if (leftQ) ham.rotateCholesky(detT, rotChol);
};

void RHF::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet) 
{
  sampleDet[0] = det;
  sampleDet[1] = det;
};


std::array<std::complex<double>, 2> RHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) 
{ 
  complex<double> overlap = (detT * psi[0]).determinant() * (detT * psi[1]).determinant();
  complex<double> ene = ham.ecore;
  
  // calculate theta and green
  matPair theta, green;
  theta[0] = psi[0] * (detT * psi[0]).inverse();
  theta[1] = psi[1] * (detT * psi[1]).inverse();
  green[0] = theta[0] * detT;
  green[1] = theta[1] * detT;

  // one body part
  ene += green[0].cwiseProduct(ham.h1).sum() + green[1].cwiseProduct(ham.h1).sum();

  // two body part
  MatrixXcd fup = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  MatrixXcd fdn = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  for (int i = 0; i < ham.nchol; i++) {
    fup.noalias() = rotChol[i] * theta[0];
    fdn.noalias() = rotChol[i] * theta[1];
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};


std::array<std::complex<double>, 2> RHF::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham) 
{ 
  complex<double> overlap = (detT * psi).determinant();
  overlap *= overlap;
  complex<double> ene = ham.ecore;
  
  // calculate theta and green
  MatrixXcd theta, green;
  theta = psi * (detT * psi).inverse();
  green = theta * detT;

  // one body part
  ene += 2. * green.cwiseProduct(ham.h1).sum();

  // two body part
  MatrixXcd f = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  for (int i = 0; i < ham.nchol; i++) {
    f.noalias() = rotChol[i] * theta;
    complex<double> c = f.trace();
    ene += (2. * c * c - f.cwiseProduct(f.transpose()).sum());
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};