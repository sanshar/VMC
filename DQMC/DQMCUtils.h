#ifndef DQMCUtils_HEADER_H
#define DQMCUtils_HEADER_H
#include <Eigen/Dense>
#include <utility>
#include <vector>

void binning(Eigen::VectorXcd& samples, Eigen::VectorXd& stdDev, Eigen::VectorXi binSizes);

Eigen::MatrixXcd matExp(const Eigen::MatrixXcd& mat, const int order = -1);

void orthogonalize(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& rn, std::complex<double>& orthoFac);

std::complex<double> prepJastrowHS(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& ref, 
    std::vector<std::pair<Eigen::VectorXcd, Eigen::VectorXcd>>& hsOperators, 
    std::pair<Eigen::VectorXcd, Eigen::VectorXcd>& oneBodyOperator);

std::complex<double> prepJastrowHS1(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& ref, 
    std::vector<std::pair<Eigen::VectorXcd, Eigen::VectorXcd>>& hsOperators, 
    std::pair<Eigen::VectorXcd, Eigen::VectorXcd>& oneBodyOperator);

std::complex<double> prepPropagatorHS(std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& ref, 
    std::vector<Eigen::MatrixXd>& chol, 
    std::vector<std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>>& hsOperators, 
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd>& oneBodyOperator);

void printEnergy(std::vector<std::complex<double>>& numMean, std::vector<std::complex<double>>& denomMean, std::vector<std::complex<double>>& denomAbsMean, std::vector<int>& eneSteps, double dt,
                    double eneTime, double propTime);

#endif
