#include <vector>
#include "LocalEnergy.h"
#include "relLocalEnergy.h"
#include "Determinants.h"
#include "relDeterminants.h"
#include "relWorkingArray.h"
#include "global.h"
#include "input.h"
#include "igl/slice.h"
#include "stan/math.hpp"
#include "Complex.h"

/*
//functions for jastrow vars
template<typename T>
void BuildJastrowVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, int &numVars)
{
    //pull jastrows from vars
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    J.setZero(2 * norbs, 2 * norbs);
    for (int i = 0; i < 2 * norbs; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            J(i, j) = vars[numVars];
            numVars++;
        }
    }
    //intermediates for efficient evaluation
    std::vector<int> open, closed;
    D.getOpenClosed(open, closed);
    Jmid.setZero(2 * norbs);
    for (int i = 0; i < 2 * norbs; i++)
    {
        Jmid[i] = J(i,i);
        for (int j = 0; j < closed.size(); j++)
        {
            if (closed[j] != i)
                Jmid[i] *= J(std::max(i, closed[j]), std::min(i, closed[j]));
        }
    }
}
template
void BuildJastrowVars(const Eigen::Matrix<double, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<double, Eigen::Dynamic, 1> &Jmid, int &numVars);
template
void BuildJastrowVars(const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> &Jmid, int &numVars);

//functions for hf vars
void concatenateGhf(const vector<int>& v1, const vector<int>& v2, vector<int>& result)
{
  int norbs = Determinant::norbs;
  result.clear();
  result = v1;
  result.insert(result.end(), v2.begin(), v2.end());    
  for (int j = v1.size(); j < v1.size() + v2.size(); j++)
    result[j] += norbs;
}

template<typename T>
void BuildOrbitalVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, std::array<Complex<T>, 2> &thetaDet, std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R, int &numVars)
{
    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;

    //pull orbitals from vars
    std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> HF;
    if (schd.hf == "rhf")
    {
        int size = norbs;
        HF[0].resize(size, size);
        HF[1].resize(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                HF[0](i, j).real(vars[numVars + 2 * i * norbs + 2 * j]);
                HF[0](i, j).imag(vars[numVars + 2 * i * norbs + 2 * j + 1]);
                HF[1](i, j).real(vars[numVars + 2 * i * norbs + 2 * j]);
                HF[1](i, j).imag(vars[numVars + 2 * i * norbs + 2 * j + 1]);
            }
        }
        numVars += 2 * norbs * norbs;
    }
    else if (schd.hf == "uhf")
    {
        int size = norbs;
        HF[0].resize(size, size);
        HF[1].resize(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                HF[0](i, j).real(vars[numVars + 2 * i * norbs + 2 * j]);
                HF[0](i, j).imag(vars[numVars + 2 * i * norbs + 2 * j + 1]);
                HF[1](i, j).real(vars[numVars + 2 * norbs * norbs + 2 * i * norbs + 2 * j]);
                HF[1](i, j).imag(vars[numVars + 2 * norbs * norbs + 2 * i * norbs + 2 * j + 1]);
            }
        }
        numVars += 2 * 2 * norbs * norbs;
    }
    else if (schd.hf == "ghf")
    {
        int size = 2 * norbs;
        HF[0].resize(size, size);
        HF[1].resize(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                HF[0](i, j).real(vars[numVars + 4 * i * norbs + 2 * j]);
                HF[0](i, j).imag(vars[numVars + 4 * i * norbs + 2 * j + 1]);
                HF[1](i, j).real(vars[numVars + 4 * i * norbs + 2 * j]);
                HF[1](i, j).imag(vars[numVars + 4 * i * norbs + 2 * j + 1]);
            }
        }
        numVars += 2 * 4 * norbs * norbs;
    }
    
    //hartree fock reference
    std::array<std::vector<int>, 2> closedOrbsRef;
    if (schd.hf == "rhf" || schd.hf == "uhf")
    {
        for (int i = 0; i < nalpha; i++)
        {
            closedOrbsRef[0].push_back(i);
        }
        for (int i = 0; i < nbeta; i++)
        {
            closedOrbsRef[1].push_back(i);
        }
    }
    else if (schd.hf == "ghf")
    {
        int nelec = nalpha + nbeta;
        for (int i = 0; i < nelec; i++)
        {
            closedOrbsRef[0].push_back(i);
            closedOrbsRef[1].push_back(i);
        }
    }
        
    //Calculate overlaps of reference and current determinant
    std::array<std::vector<int>, 2> closedOrbs, openOrbs;
    D.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    if (schd.hf == "rhf" || schd.hf == "uhf")
    {
        for (int sz = 0; sz < 2; sz++)
        {
            Eigen::Map<Eigen::VectorXi> rowClosed(&closedOrbs[sz][0], closedOrbs[sz].size());
            Eigen::Map<Eigen::VectorXi> colClosed(&closedOrbsRef[sz][0], closedOrbsRef[sz].size());

            Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> O;
            igl::slice(HF[sz], rowClosed, colClosed, O);

            Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> OInv;
            //Eigen::FullPivLU<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>> lua(O);
            //OInv = lua.inverse();
            //thetaDet[sz] = lua.determinant();
            OInv = stan::math::inverse(O);
            thetaDet[sz] = stan::math::determinant(O);

            Eigen::Map<Eigen::VectorXi> rowOpen(&openOrbs[sz][0], openOrbs[sz].size());
            Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> HfO;
            igl::slice(HF[sz], rowOpen, colClosed, HfO);
            R[sz] = HfO * OInv;
            //R[sz] = stan::math::multiply(HfO, OInv);
        }
     }
     else if (schd.hf == "ghf")
     {
         std::vector<int> workingVec;
         concatenateGhf(closedOrbs[0], closedOrbs[1], workingVec);
         Eigen::Map<Eigen::VectorXi> rowClosed(&workingVec[0], workingVec.size());
         Eigen::Map<Eigen::VectorXi> colClosed(&closedOrbsRef[0][0], closedOrbsRef[0].size());

         Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> O;
         igl::slice(HF[0], rowClosed, colClosed, O);

         Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> OInv;
         //Eigen::FullPivLU<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>> lua(O);
         //OInv = lua.inverse();
         //thetaDet[0] = lua.determinant();
         OInv = stan::math::inverse(O);
         thetaDet[0] = stan::math::determinant(O);
         thetaDet[1].real(1.0);
         thetaDet[1].imag(0.0);

         Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> HfO;
         std::vector<int> rowVec;
         concatenateGhf(openOrbs[0], openOrbs[1], rowVec);
         Eigen::Map<Eigen::VectorXi> rowOpen(&rowVec[0], rowVec.size());
         igl::slice(HF[0], rowOpen, colClosed, HfO);
         R[0] = HfO * OInv;
         //R[0] = stan::math::multiply(HfO, OInv);
         R[1] = R[0];
     }
}
template
void BuildOrbitalVars(const Eigen::Matrix<double, Eigen::Dynamic, 1> &vars, const Determinant &D, std::array<Complex<double>, 2> &thetaDet, std::array<Eigen::Matrix<Complex<double>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R, int &numVars);
template
void BuildOrbitalVars(const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> &vars, const Determinant &D, std::array<Complex<stan::math::var>, 2> &thetaDet, std::array<Eigen::Matrix<Complex<stan::math::var>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R, int &numVars);


//functions for hf local energy evaluation
void getHfRelIndices(int i, int &relI, int a, int &relA, bool sz, const std::array<std::vector<int>, 2> &closedOrbs, const std::array<std::vector<int>, 2> &openOrbs)
{
    //std::array<vector<int>, 2> closedOrbs, openOrbs;
    //D.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    int factor = 0;
    if (schd.hf == "ghf" && sz != 0) 
        factor = 1;
    relI = std::search_n(closedOrbs[sz].begin(), closedOrbs[sz].end(), 1, i) - closedOrbs[sz].begin() + factor * closedOrbs[0].size();
    relA = std::search_n(openOrbs[sz].begin(), openOrbs[sz].end(), 1, a) - openOrbs[sz].begin() + factor * openOrbs[0].size();
    //relA = a + factor * Determinant::norbs;
}
*/


template<typename T>
T JastrowSlaterLocalEnergy(const relDeterminant &D, const relWorkingArray &work, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, const std::array<Complex<T>, 2> &thetaDet, const std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R)
{
    cout << "in JastrowSlaterLocalEnergy" << endl;
    int norbs = relDeterminant::norbs;
    int nalpha = relDeterminant::nalpha;
    int nbeta = relDeterminant::nbeta;
    std::array<vector<int>, 2> closedOrbs, openOrbs;
    D.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
    T Eloc = D.Energy(I1, I2, coreE);
    for (int l = 0; l < work.nExcitations; l++)
    {
        int ex1 = work.excitation1[l], ex2 = work.excitation2[l];
        std::complex<T> tia = work.HijElement[l];

        int i = ex1 / 2 / norbs, a = ex1 - 2 * norbs * i;
        int j = ex2 / 2 / norbs, b = ex2 - 2 * norbs * j;

        int ri, ra, rj, rb;
        int sz1, sz2;

        Complex<T>  ovlpRatio = 1.0;

        if (j == b && b == 0) //single excitations // EDIT THINK: so far only spin conserved, do we need a spin flip term ?
        {
            ovlpRatio *= Jmid[a] / Jmid[i] / J(std::max(i,a), std::min(i,a));
            if (i % 2 == 0) //alpha
                sz1 = 0; 
            else    //beta
                sz1 = 1;
            getHfRelIndices(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
            ovlpRatio *= (R[sz1](ra, ri) * thetaDet[0] * thetaDet[1]) / (thetaDet[0] * thetaDet[1]);
        }
        else //double excitations
        {
            ovlpRatio *= Jmid[a] * Jmid[b] * J(std::max(a,b), std::min(a,b)) * J(std::max(i,j), std::min(i,j)) / Jmid[i] / Jmid[j] / J(std::max(a,i), std::min(a,i)) / J(std::max(a,j), std::min(a,j)) / J(std::max(i,b), std::min(i,b)) / J(std::max(j,b), std::min(j,b));
            if (i % 2 == j % 2 && i % 2 == 0) //aa to aa
            {
                sz1 = 0;
                sz2 = 0;
            }
            else if (i % 2 == j % 2 && i % 2 == 1) //bb to bb
            {
                sz1 = 1;
                sz2 = 1;
            }
            else if (i % 2 != j % 2 && i % 2 == 0) //ab to ab
            {
                sz1 = 0;
                sz2 = 1;
            }
            else  //ba to ba // EDIT THINK: so far only spin conserved, do we need a spin flip term
            {
                sz1 = 1;
                sz2 = 0;
            }
            getHfRelIndices(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
            getHfRelIndices(j / 2, rj, b / 2, rb, sz2, closedOrbs, openOrbs); 
            if (sz1 == sz2 || schd.hf == "ghf")
            {
                ovlpRatio *= ((R[sz1](ra, ri) * R[sz1](rb, rj) - R[sz1](rb, ri) * R[sz1](ra, rj))* thetaDet[0] * thetaDet[1]) / (thetaDet[0] * thetaDet[1]);
            }
            else
            {
                ovlpRatio *= ((R[sz1](ra, ri) * R[sz2](rb, rj)) * thetaDet[0] * thetaDet[1]) / (thetaDet[0] * thetaDet[1]);
            }
        }
        std::complex<T> ovlpRatio_as_std;
        ovlpRatio_as_std.real(ovlpRatio.real());
        ovlpRatio_as_std.imag(ovlpRatio.imag());
        Eloc += (tia * ovlpRatio_as_std).real(); //EDIT DO: now real part taken
    }
    return Eloc;
}    
template
double JastrowSlaterLocalEnergy(const relDeterminant &D, const relWorkingArray &work, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<double, Eigen::Dynamic, 1> &Jmid, const std::array<Complex<double>, 2> &thetaDet, const std::array<Eigen::Matrix<Complex<double>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R);
//template
//stan::math::var JastrowSlaterLocalEnergy(const Determinant &D, const relWorkingArray &work, const Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> &Jmid, const std::array<Complex<stan::math::var>, 2> &thetaDet, const std::array<Eigen::Matrix<Complex<stan::math::var>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R);

// if also pfaffians, include here

