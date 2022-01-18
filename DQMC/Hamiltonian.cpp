#include "Hamiltonian.h"
#include "integral.h"

using namespace std;
using namespace Eigen;

// constructor
Hamiltonian::Hamiltonian(string fname) 
{
  readIntegralsCholeskyAndInitializeDeterminantStaticVariables(fname, norbs, nalpha, nbeta, ecore, h1, h1Mod, chol);
  nchol = chol.size();
  floattenCholesky();
};


void Hamiltonian::setNchol(int pnchol) 
{
  nchol = pnchol;
};


// rotate cholesky
void Hamiltonian::rotateCholesky(std::array<Eigen::MatrixXd, 2>& phiT, std::vector<std::array<Eigen::MatrixXd, 2>>& rotChol) 
{
  for (int i = 0; i < chol.size(); i++) {
    std::array<Eigen::MatrixXd, 2> rot;
    rot[0] = phiT[0] * chol[i][0];
    rot[1] = phiT[1] * chol[i][1];
    rotChol.push_back(rot);
  }
};


// flatten and convert to float
//void Hamiltonian::floattenCholesky(std::vector<Eigen::MatrixXf>& floatChol)
//void Hamiltonian::floattenCholesky(std::vector<vector<float>>& floatChol)
void Hamiltonian::floattenCholesky()
{
  //for (int n = 0; n < chol.size(); n++) floatChol.push_back(chol[n].cast<float>());
  for (int n = 0; n < chol.size(); n++) {
    std::array<vector<float>, 2> cholVec;
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j <= i; j++) {
        cholVec[0].push_back(float(chol[n][0](i, j)));
        cholVec[1].push_back(float(chol[n][1](i, j)));
      }
    }
    floatChol.push_back(cholVec);
  }
};
