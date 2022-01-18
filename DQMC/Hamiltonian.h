#ifndef Hamiltonian_HEADER_H
#define Hamiltonian_HEADER_H
#include <string>
#include <vector>
#include <Eigen/Dense>

// cholesky vectors
class Hamiltonian {
  public:
    std::array<Eigen::MatrixXd, 2> h1, h1Mod;
    std::vector<std::array<Eigen::MatrixXd, 2>> chol;
    std::vector<std::array<std::vector<float>, 2>> floatChol;
    double ecore;
    int norbs, nalpha, nbeta, nchol;

    // constructor
    Hamiltonian(std::string fname);

    void setNchol(int pnchol);

    // rotate cholesky
    void rotateCholesky(std::array<Eigen::MatrixXd, 2>& phi, std::vector<std::array<Eigen::MatrixXd, 2>>& rotChol);

    // flatten and convert to float
    //void floattenCholesky(std::vector<Eigen::MatrixXf>& floatChol);
    //void floattenCholesky(std::vector<std::vector<float>>& floatChol);
    void floattenCholesky();
};
#endif
