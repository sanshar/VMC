#include "rCorrelatedWavefunction.h"
#include "rPseudopotential.h"
#include "rMultiSlater.h"
#include "rSlater.h"
#include "rBFSlater.h"
#include "rJastrow.h"
#include "global.h"
#include "input.h"
#include <boost/math/special_functions/legendre.hpp>

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::updateOptVariables(Eigen::VectorXd &v) 
{
  if (schd.optimizeCps) {
    corr.updateVariables(v);
  }
  if (schd.optimizeOrbs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.updateVariables(v.tail(getNumRefVariables()));
  }
}

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::getOptVariables(Eigen::VectorXd &v) const
{
  v.setZero(getNumOptVariables());
  if (schd.optimizeCps) {
    corr.getVariables(v);
  }
  if (schd.optimizeOrbs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.getVariables(vtail);
  }
}

template<>
long rCorrelatedWavefunction<rJastrow, rSlater>::getNumOptVariables() const
{
  long numVars = 0;
  if (schd.optimizeCps) numVars += getNumJastrowVariables();
  if (schd.optimizeOrbs) numVars += getNumRefVariables();
  return numVars;
}

template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::HamOverlap(rWalker<rJastrow, rSlater>& walk,
                                                              Eigen::VectorXd& gradRatio,
                                                              Eigen::VectorXd& hamRatio) const
{
  gradRatio.setZero(getNumOptVariables());
  hamRatio.setZero(getNumOptVariables());
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  std::complex<double> thetaDet = walk.refHelper.thetaDet[0][0] * walk.refHelper.thetaDet[0][1];
  std::array<MatrixXcd, 2> thetaInv = walk.refHelper.thetaInv;
  std::complex<double> i(0.0, 1.0);

  //true and complex local energy
  double Eloc;
  std::complex<double> cEloc;

  double potentialij = 0.0, potentiali = 0.0, potentiali_ppl = 0.0, potentiali_ppnl = 0.0, potentialN = 0.0;
  std::complex<double> cpotentiali_ppnl = 0.0;

  //get potential
  for (int i=0; i<nelec; i++)
    for (int j=i+1; j<nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  //pseudopotential
  const Pseudopotential &pp = *schd.pseudo;
  std::vector<std::vector<double>> viq; //matrix elements of nonlocal potential for every quadrature point [elec][quad]
  std::vector<std::vector<Vector3d>> riq; //coordinates corresponding to above matrix elements
  std::vector<std::vector<double>> Yiq; //reference overlap ratio for above coordinates
  std::vector<std::vector<std::complex<double>>> cYiq; //complex overlap ratio
  std::vector<std::vector<double>> Jiq; //jastrow overlap ratio for above coordinates
  std::vector<std::vector<VectorXd>> AOiqn; // \phi_n (q) atomic orbitals evaluated at every quadrature point [elec][q](n)
  std::vector<std::vector<VectorXcd>> MOiqm; // \psi_m (q) molecular orbitals evaluated at every quadrature point [elec][q](m)
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    potentiali_ppl += pp.localPotential(walk.d);

    //nonlocal potential
    pp.nonLocalPotential(walk.d, viq, riq); 

    //calculate reference and jastrow overlap ratios
    //this basically populates all the above tensors for later use in local gradients and energy
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> Yq;
      std::vector<std::complex<double>> cYq;
      std::vector<double> Jq;
      std::vector<VectorXd> AOqn;
      std::vector<VectorXcd> MOqm;

      //options for rhf/uhf
      int sz = i < nalpha ? 0 : 1;
      int nmo = i < nalpha ? nalpha : nbeta;

      for (int q = 0; q < riq[i].size(); q++)
      {
        //jastrow ratio
        Jq.push_back(walk.corrHelper.OverlapRatio(i, riq[i][q], corr, walk.d));
        
        //reference ratio
        //eval basis
        schd.basis->eval(riq[i][q], &walk.refHelper.aoValues[0]);

        //atomic orbitals
        VectorXd basis(norbs);
        for (int j = 0; j < norbs; j++) { basis(j) = walk.refHelper.aoValues[j]; }
        AOqn.push_back(basis);

        //ratio
        std::complex<double> factor = 0.0;
        if (schd.hf == "ghf")
        {
          //molecular orbital
          VectorXcd row = VectorXd::Zero(nelec);
          for (int mo = 0; mo < nelec; mo++) {
            for (int j = 0; j < norbs; j++) {
              int J = i < nalpha ? j : j + norbs;
              row(mo) += walk.refHelper.aoValues[j] * ref.getHforbs(0)(J, mo);
            }
          }
          MOqm.push_back(row);
          factor = row.transpose() * walk.refHelper.thetaInv[0].col(i);
        }
        else //rhf/ufh
        {
          //molecular orbital
          VectorXcd row = VectorXd::Zero(nmo);
          for (int mo = 0; mo < nmo; mo++) {
            for (int j = 0; j < norbs; j++) {
              row(mo) += walk.refHelper.aoValues[j] * ref.getHforbs(sz)(j, mo);
            }
          }
          MOqm.push_back(row);
          factor = row.transpose() * walk.refHelper.thetaInv[sz].col(i - sz * walk.d.nalpha);
        }

        //update tensors
        cYq.push_back(factor);
        Yq.push_back((factor * thetaDet).real() / thetaDet.real());

      }//q

      //update tensors
      Jiq.push_back(Jq);
      cYiq.push_back(cYq);
      Yiq.push_back(Yq);
      AOiqn.push_back(AOqn);
      MOiqm.push_back(MOqm);
    }//nelec

    //calculate nonlocal potential
    for (int i = 0; i < nelec; i++)
    {
      for (int q = 0; q < riq[i].size(); q++)
      {
        potentiali_ppnl += viq[i][q] * Jiq[i][q] * Yiq[i][q];
        cpotentiali_ppnl += viq[i][q] * Jiq[i][q] * cYiq[i][q];
      }
    }
  }

  double kinetic = 0.0; 
  std::complex<double> ckinetic = 0.0;
  {
    MatrixXcd Bij = walk.refHelper.Laplacian; //i = nelec , j = norbs
    for (int i = 0; i < nelec; i++) {
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,0) * walk.refHelper.Gradient[0].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,1) * walk.refHelper.Gradient[1].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,2) * walk.refHelper.Gradient[2].row(i);
    }

    for (int i = 0; i < nelec; i++) {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bij.row(i) * thetaInv[0].col(i); }
      else {
        if (i < nalpha) { factor = Bij.row(i).head(nalpha) * thetaInv[0].col(i); }
        else { factor = Bij.row(i).tail(nbeta) * thetaInv[1].col(i - nalpha); }
      }
      kinetic += (thetaDet * factor).real() / thetaDet.real();
      kinetic += walk.corrHelper.LaplaceRatio[i];
      ckinetic += factor;
      ckinetic += walk.corrHelper.LaplaceRatio[i];
    }
  }
  Eloc = -0.5 * kinetic + potentialij + potentiali + potentiali_ppl + potentiali_ppnl + potentialN; 
  cEloc = -0.5 * ckinetic + potentialij + potentiali + potentiali_ppl + cpotentiali_ppnl + potentialN; 
  //cout << "hamOvlp" << endl;
  //cout << -0.5 * kinetic << " " << potentialij << " " << potentiali << " " << potentiali_ppl << " " << potentiali_ppnl << " " << potentialN; 
  //cout << endl << endl;
 
  int numVars = 0;
  VectorXd CPSgradRatio = VectorXd::Zero(0);
  VectorXd CPShamRatio = VectorXd::Zero(0);
  //*********calculate gradRatio and hamRatio for jastrows
  if (schd.optimizeCps)
  {
      CPSgradRatio = VectorXd::Zero(getNumJastrowVariables());
      CPShamRatio = VectorXd::Zero(getNumJastrowVariables());

      numVars += corr._params.size();
      for (int j = 0; j < corr._params.size(); j++)
      {
          CPSgradRatio[j] = walk.corrHelper.ParamValues[j];
          VectorXcd Bij;
          for (int i = 0; i < nelec; i++)
          {
              Bij  = -walk.corrHelper.ParamGradient[0](i,j) * walk.refHelper.Gradient[0].row(i);
              Bij += -walk.corrHelper.ParamGradient[1](i,j) * walk.refHelper.Gradient[1].row(i);
              Bij += -walk.corrHelper.ParamGradient[2](i,j) * walk.refHelper.Gradient[2].row(i);
        
              std::complex<double> factor = 0.0;
              if (schd.hf == "ghf") { factor = Bij.transpose() * thetaInv[0].col(i); }
              else {
                if (i < nalpha) { factor = Bij.transpose().head(nalpha) * thetaInv[0].col(i); }
                else { factor = Bij.transpose().tail(nbeta) * thetaInv[1].col(i - nalpha); }
              }
              CPShamRatio[j] += (thetaDet * factor).real() / thetaDet.real();
              CPShamRatio[j] += -0.5*(walk.corrHelper.ParamLaplacian(i, j) +
                               2.*walk.corrHelper.GradRatio(i,0)*walk.corrHelper.ParamGradient[0](i,j)+
                               2.*walk.corrHelper.GradRatio(i,1)*walk.corrHelper.ParamGradient[1](i,j)+
                               2.*walk.corrHelper.GradRatio(i,2)*walk.corrHelper.ParamGradient[2](i,j));
          }
      }

      if (pp.size())
      {
        for (int i = 0; i < nelec; i++)
        {
          for (int q = 0; q < riq[i].size(); q++)
          {
            //corr ratio and param gradient
            VectorXd g = VectorXd::Zero(getNumJastrowVariables());
            double corrRatio = walk.corrHelper.OverlapRatioAndParamGradient(i, riq[i][q], corr, walk.d, g);
            //gradient element
            CPShamRatio += viq[i][q] * Yiq[i][q] * corrRatio * g;
          }
        }
      }

      CPShamRatio[corr.EEsameSpinIndex] = 0.0;
      CPShamRatio[corr.EEoppositeSpinIndex] = 0.0;
      if (schd.noENCusp || schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { CPShamRatio[corr.ENIndex + I * corr.Qmax] = 0.0; }
      CPSgradRatio[corr.EEsameSpinIndex] = 0.0;
      CPSgradRatio[corr.EEoppositeSpinIndex] = 0.0;
      if (schd.noENCusp || schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { CPSgradRatio[corr.ENIndex + I * corr.Qmax] = 0.0; }
      CPShamRatio += Eloc * CPSgradRatio;
  }
   
  //*********calculate the hamoverlap for orbitals
  VectorXd RefgradRatio = VectorXd::Zero(0);
  VectorXd RefhamRatio = VectorXd::Zero(0);
  VectorXcd RefGradcOvlp = VectorXcd::Zero(0);
  VectorXcd RefGradcEloc = VectorXcd::Zero(0);

  if (schd.optimizeOrbs) {
    RefgradRatio = VectorXd::Zero(getNumRefVariables());
    RefhamRatio = VectorXd::Zero(getNumRefVariables());
    RefGradcOvlp = VectorXcd::Zero(getNumRefVariables());
    RefGradcEloc = VectorXcd::Zero(getNumRefVariables());

    MatrixXd AoRi = walk.refHelper.AO;
    MatrixXd AOLaplacian = walk.refHelper.AOLaplacian;
    MatrixXd AOGradx = walk.refHelper.AOGradient[0];
    MatrixXd AOGrady = walk.refHelper.AOGradient[1];
    MatrixXd AOGradz = walk.refHelper.AOGradient[2]; 

    MatrixXcd Laplacian = walk.refHelper.Laplacian;
    MatrixXcd Gradx = walk.refHelper.Gradient[0];
    MatrixXcd Grady = walk.refHelper.Gradient[1];
    MatrixXcd Gradz = walk.refHelper.Gradient[2];

    for (int mo = 0; mo < nelec; mo++) {
      Gradx.row(mo) *= walk.corrHelper.GradRatio(mo,0);
      Grady.row(mo) *= walk.corrHelper.GradRatio(mo,1);
      Gradz.row(mo) *= walk.corrHelper.GradRatio(mo,2);
      AOGradx.row(mo) *= walk.corrHelper.GradRatio(mo,0);
      AOGrady.row(mo) *= walk.corrHelper.GradRatio(mo,1);
      AOGradz.row(mo) *= walk.corrHelper.GradRatio(mo,2);
    }

    //for pseudopotential gradients
    int nao = schd.hf == "ghf" ? 2*norbs : norbs; //num atomic orbitals
    MatrixXcd Bnl = MatrixXd::Zero(nelec, nelec); //Bnl(elec, mo) = Vnl * DetMatrix     
    MatrixXd AOBnl = MatrixXd::Zero(nelec, nao);  //AOBnl(elec, ao) = \partial_{mocoeff} Vnl * DetMatrix    
    if (pp.size())
    {
      for (int i = 0; i < nelec; i++)
      {
        for (int q = 0; q < riq[i].size(); q++)
        {
          //atomic orbitals
          for (int j = 0; j < norbs; j++)
          {
            int J = i < nalpha ? j : j + norbs;
            if (schd.hf == "ghf") { AOBnl(i, J) += viq[i][q] * Jiq[i][q] * AOiqn[i][q][j]; }
            else { AOBnl(i, j) += viq[i][q] * Jiq[i][q] * AOiqn[i][q][j]; }
          }

          //molecular orbitals
          int nmo = i < nalpha ? nalpha : nbeta; //for hf
          if (schd.hf == "ghf")
          {
            for (int j = 0; j < nelec; j++)
            {
              Bnl(i, j) += viq[i][q] * Jiq[i][q] * MOiqm[i][q][j];
            }
          }
          else //rhf/uhf
          {
            for (int j = 0; j < nmo; j++)
            {
              int J = i < nalpha ? j : j + nalpha;
              Bnl(i, J) += viq[i][q] * Jiq[i][q] * MOiqm[i][q][j];
            }
          }

        }//q
      }//nelec
    }//if
 
    if (schd.hf == "ghf")
    {
      MatrixXcd X = thetaInv[0] * Laplacian * thetaInv[0];
      MatrixXcd Xgx = thetaInv[0] * Gradx * thetaInv[0];
      MatrixXcd Xgy = thetaInv[0] * Grady * thetaInv[0];
      MatrixXcd Xgz = thetaInv[0] * Gradz * thetaInv[0];
      MatrixXcd Xnl;
      if (pp.size()) { Xnl = thetaInv[0] * Bnl * thetaInv[0]; }

      for (int mo = 0; mo < nelec; mo++) { 
        for (int orb = 0; orb < 2*norbs; orb++) {
          //nonlocal potential contribution 
          if (pp.size())
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOBnl.col(orb);
            std::complex<double> t2 = Xnl.row(mo) * AoRi.col(orb);
            std::complex<double> factor = t1 - t2;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
          }

          //laplacian contribution
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOLaplacian.col(orb);
            std::complex<double> t2 = X.row(mo) * AoRi.col(orb);
            std::complex<double> factor = -0.5 * (t1 - t2);
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
          }

          //grad contribution
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGradx.col(orb);
            std::complex<double> t2 = Xgx.row(mo) * AoRi.col(orb);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGrady.col(orb);
            std::complex<double> t2 = Xgy.row(mo) * AoRi.col(orb);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGradz.col(orb);
            std::complex<double> t2 = Xgz.row(mo) * AoRi.col(orb);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
          }

          {
            std::complex<double> factor = thetaInv[0].row(mo) * AoRi.col(orb);
            RefGradcOvlp[numDets + 2*orb * nelec + 2*mo] += factor;
            RefGradcOvlp[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
            RefgradRatio[numDets + 2*orb * nelec + 2*mo] += (factor * thetaDet).real() / thetaDet.real();        
            if (schd.ifComplex) RefgradRatio[numDets + 2*orb * nelec + 2*mo + 1] += (i * factor * thetaDet).real() / thetaDet.real();
          }
        } 
      }
    }
    else
    { //rhf/uhf

      //alpha
      MatrixXcd X = thetaInv[0] * Laplacian.topLeftCorner(nalpha, nalpha) * thetaInv[0];
      MatrixXcd Xgx = thetaInv[0] * Gradx.topLeftCorner(nalpha, nalpha) * thetaInv[0];
      MatrixXcd Xgy = thetaInv[0] * Grady.topLeftCorner(nalpha, nalpha) * thetaInv[0];
      MatrixXcd Xgz = thetaInv[0] * Gradz.topLeftCorner(nalpha, nalpha) * thetaInv[0];
      MatrixXcd Xnl;
      if (pp.size()) { Xnl = thetaInv[0] * Bnl.topLeftCorner(nalpha, nalpha) * thetaInv[0]; }

      for (int mo = 0; mo < nalpha; mo++) { 
        for (int orb = 0; orb < norbs; orb++) {
          //nonlocal potential contribution 
          if (pp.size())
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOBnl.col(orb).head(nalpha);
            std::complex<double> t2 = Xnl.row(mo) * AoRi.col(orb).head(nalpha);
            std::complex<double> factor = t1 - t2;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
          }

          //laplacian contribution
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOLaplacian.col(orb).head(nalpha);
            std::complex<double> t2 = X.row(mo) * AoRi.col(orb).head(nalpha);
            std::complex<double> factor = -0.5 * (t1 - t2);
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
          }

          //grad contribution
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGradx.col(orb).head(nalpha);
            std::complex<double> t2 = Xgx.row(mo) * AoRi.col(orb).head(nalpha);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGrady.col(orb).head(nalpha);
            std::complex<double> t2 = Xgy.row(mo) * AoRi.col(orb).head(nalpha);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[0].row(mo) * AOGradz.col(orb).head(nalpha);
            std::complex<double> t2 = Xgz.row(mo) * AoRi.col(orb).head(nalpha);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcEloc[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
          }

          {
            std::complex<double> factor = thetaInv[0].row(mo) * AoRi.col(orb).head(nalpha);
            RefGradcOvlp[numDets + 2*orb * nalpha + 2*mo] += factor;
            RefGradcOvlp[numDets + 2*orb * nalpha + 2*mo + 1] += i * factor;
            RefgradRatio[numDets + 2*orb * nalpha + 2*mo] += (factor * thetaDet).real() / thetaDet.real();        
            if (schd.ifComplex) RefgradRatio[numDets + 2*orb * nalpha + 2*mo + 1] += (i * factor * thetaDet).real() / thetaDet.real();
          }
        } 
      } 

      //beta
      int shift = 0;
      if (schd.hf == "uhf") { shift = 2*nalpha*norbs; }

      X = thetaInv[1] * Laplacian.bottomRightCorner(nbeta, nbeta) * thetaInv[1];
      Xgx = thetaInv[1] * Gradx.bottomRightCorner(nbeta, nbeta) * thetaInv[1];
      Xgy = thetaInv[1] * Grady.bottomRightCorner(nbeta, nbeta) * thetaInv[1];
      Xgz = thetaInv[1] * Gradz.bottomRightCorner(nbeta, nbeta) * thetaInv[1];
      if (pp.size()) { Xnl = thetaInv[1] * Bnl.bottomRightCorner(nbeta, nbeta) * thetaInv[1]; }

      for (int mo = 0; mo < nbeta; mo++) { 
        for (int orb = 0; orb < norbs; orb++) {
          //nonlocal potential contribution 
          if (pp.size())
          {
            std::complex<double> t1 = thetaInv[1].row(mo) * AOBnl.col(orb).tail(nbeta);
            std::complex<double> t2 = Xnl.row(mo) * AoRi.col(orb).tail(nbeta);
            std::complex<double> factor = t1 - t2;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
          }

          //laplacian contribution
          {
            std::complex<double> t1 = thetaInv[1].row(mo) * AOLaplacian.col(orb).tail(nbeta);
            std::complex<double> t2 = X.row(mo) * AoRi.col(orb).tail(nbeta);
            std::complex<double> factor = -0.5 * (t1 - t2);
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
          }

          //grad contribution
          {
            std::complex<double> t1 = thetaInv[1].row(mo) * AOGradx.col(orb).tail(nbeta);
            std::complex<double> t2 = Xgx.row(mo) * AoRi.col(orb).tail(nbeta);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[1].row(mo) * AOGrady.col(orb).tail(nbeta);
            std::complex<double> t2 = Xgy.row(mo) * AoRi.col(orb).tail(nbeta);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
          }
          {
            std::complex<double> t1 = thetaInv[1].row(mo) * AOGradz.col(orb).tail(nbeta);
            std::complex<double> t2 = Xgz.row(mo) * AoRi.col(orb).tail(nbeta);
            std::complex<double> factor = -(t1 - t2);
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcEloc[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
          }

          {
            std::complex<double> factor = thetaInv[1].row(mo) * AoRi.col(orb).tail(nbeta);
            RefGradcOvlp[numDets + shift + 2*orb * nbeta + 2*mo] += factor;
            RefGradcOvlp[numDets + shift + 2*orb * nbeta + 2*mo + 1] += i * factor;
            RefgradRatio[numDets + shift + 2*orb * nbeta + 2*mo] += (factor * thetaDet).real() / thetaDet.real();        
            if (schd.ifComplex) RefgradRatio[numDets + shift + 2*orb * nbeta + 2*mo + 1] += (i * factor * thetaDet).real() / thetaDet.real();
          }
        } 
      }
    }
    RefhamRatio = ((cEloc * RefGradcOvlp + RefGradcEloc) * thetaDet).real() / thetaDet.real();
  } //opt orbs
  hamRatio << CPShamRatio, RefhamRatio;
  gradRatio << CPSgradRatio, RefgradRatio;
  return Eloc;
}


template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::enforceCusp() {
  //return;
  int natom = schd.Ncharge.size();
  int norbs = schd.basis->getNorbs();
  MatrixXd Pmatrix = MatrixXd::Zero(norbs, natom);

  //it only makes sense to enforce cusp with slater basis
  slaterBasis &basis = dynamic_cast<slaterBasis&>(*schd.basis);
  
  vector<double> aoValues(norbs);
  
  for (int atom = 0; atom<schd.Ncharge.size(); atom++) {
    Vector3d coord = schd.Ncoords[atom];
    double charge = schd.Ncharge[atom];
    
    basis.eval(coord, &aoValues[0]);
  
    for (int ao = 0; ao <norbs; ao++) 
      Pmatrix(ao, atom) += charge*aoValues[ao];

    //add cusp
    int nbasisfns = 0;
    for (int n=0; n<atom; n++)
      nbasisfns += basis.atomicBasis[atom].norbs;
    
    slaterBasisOnAtom& atomBasis = basis.atomicBasis[atom];
    for (int b=0; b<atomBasis.exponents.size(); b++) {
      if (atomBasis.NL[2*b] == 1 && atomBasis.NL[2*b+1] == 0) { //1s
        Pmatrix(nbasisfns, atom) += -atomBasis.exponents[b]*atomBasis.radialNorm[b];
        nbasisfns ++;
      }
      else if (atomBasis.NL[2*b] == 2 && atomBasis.NL[2*b+1] == 0) { //2s
        Pmatrix(nbasisfns, atom) += atomBasis.radialNorm[b];
        nbasisfns ++;
      }
      else { //all others
        int l = atomBasis.NL[2*b+1];
        nbasisfns += (l+1)*(l+2)/2;
      }
    }
  }

  //Now form the projector
  MatrixXd Wbc = Pmatrix.transpose()*Pmatrix;
  MatrixXd Wbcinv = Wbc.inverse();
  MatrixXd Projector = MatrixXd::Identity(norbs, norbs);
  
  for (int atom1 = 0; atom1<natom; atom1++)
    for (int atom2 = 0; atom2<natom; atom2++)
      Projector -= Wbcinv(atom1, atom2)*(Pmatrix.col(atom1)*Pmatrix.transpose().row(atom2));

  if (schd.hf == "ghf") {
    ref.HforbsA.block(0,0,norbs, 2*norbs) = Projector * ref.HforbsA.block(0,0,norbs, 2*norbs);
    ref.HforbsA.block(norbs,0,norbs, 2*norbs) = Projector * ref.HforbsA.block(norbs,0,norbs, 2*norbs);
  }
  else {
    ref.HforbsA = Projector * ref.HforbsA;
    ref.HforbsB = Projector * ref.HforbsB;
  }

}

template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(rWalker<rJastrow, rSlater>& walk) const {
  int norbs = Determinant::norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();
  
  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  double potentialij = 0.0, potentiali = 0.0, potentiali_ppl = 0.0, potentiali_ppnl = 0.0, potentialN = 0.0;

  //get potential
  for (int i=0; i<walk.d.nelec; i++)
    for (int j=i+1; j<walk.d.nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  //pseudopotential
  const Pseudopotential &pp = *schd.pseudo;
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    potentiali_ppl += pp.localPotential(walk.d);

    //nonlocal potential
    std::vector<std::vector<double>> viq;
    std::vector<std::vector<Vector3d>> riq;
    pp.nonLocalPotential(walk.d, viq, riq);
    
    for (int i = 0; i < nelec; i++)
    {
      for (int q = 0; q < riq[i].size(); q++)
      {
        potentiali_ppnl += viq[i][q] * getOverlapFactor(i, riq[i][q], walk);
      }
    }
  }
  
  double kinetic = 0.0;  
  {
    MatrixXcd Bij = walk.refHelper.Laplacian; //i = nelec , j = norbs

    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,0) * walk.refHelper.Gradient[0].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,1) * walk.refHelper.Gradient[1].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,2) * walk.refHelper.Gradient[2].row(i);
    }

    std::complex<double> DetFactor = walk.refHelper.thetaDet[0][0] * walk.refHelper.thetaDet[0][1];
    for (int i=0; i<nelec; i++) {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bij.row(i) * walk.refHelper.thetaInv[0].col(i); }
      else
      {
        if (i < walk.d.nalpha) { factor = Bij.row(i).head(walk.d.nalpha) * walk.refHelper.thetaInv[0].col(i); }
        else { factor = Bij.row(i).tail(walk.d.nbeta) * walk.refHelper.thetaInv[1].col(i - walk.d.nalpha); }
      }
      kinetic += (DetFactor * factor).real() / DetFactor.real();
      kinetic += walk.corrHelper.LaplaceRatio[i];
    }
  }
  //cout << "rHam" << endl;
  //cout << -0.5*(kinetic) << " " << potentialij << " " << potentiali << " " << potentiali_ppl << " " << potentiali_ppnl << " " << potentialN << endl;
  //cout << endl;
  return -0.5*(kinetic) + potentialij + potentiali + potentiali_ppl + potentiali_ppnl + potentialN; 
}

//this is used only in the dmc algorithm, generates tmoves with the corresponding effective hamiltonian
//size-consistent T moves
//J. Chem. Phys. 132, 154113 (2010)
double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(rWalker<rJastrow, rSlater>& walk, double& T, double& Vij, double& ViI, double& Vnl, double& VIJ, std::vector<std::vector<double>> &Viq, std::vector<std::vector<Vector3d>> &Riq) const {
  int norbs = Determinant::norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  //get potential
  Vij = 0.0, ViI = 0.0, VIJ = 0.0, Vnl = 0.0;

  for (int i=0; i<walk.d.nelec; i++)
    for (int j=i+1; j<walk.d.nelec; j++) {
      Vij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      ViI -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      VIJ += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  //pseudopotential
  const Pseudopotential &pp = *schd.pseudo;
  double Vminus = 0.0, Vplus = 0.0;
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    ViI += pp.localPotential(walk.d);

    //nonlocal potential
    std::vector<std::vector<double>> viq;
    std::vector<std::vector<Vector3d>> riq;
    pp.nonLocalPotential(walk.d, viq, riq);
    
    //calculate wavefunction overlap ratios with riq;
    std::vector<std::vector<double>> Yiq;
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> Yq;
      for (int q = 0; q < riq[i].size(); q++)
      {
        double ovlpRatio = getOverlapFactor(i, riq[i][q], walk);
        Yq.push_back(ovlpRatio);
      }
      Yiq.push_back(Yq);
    }

    //calculate nonlocal potential and matrix elements for t moves
    Viq.clear();
    Riq.clear();
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> Vq;
      std::vector<Vector3d> Rq;
      for (int q = 0; q < riq[i].size(); q++)
      {
        //matrix element
        double vxpx = viq[i][q] * Yiq[i][q];

        if (vxpx < 0.0)
        {
          Vminus += vxpx;
          Vq.push_back(vxpx);
          Rq.push_back(riq[i][q]);
        }
        else
        {
          Vplus += vxpx;
        }
      }
      //update tensors
      Viq.push_back(Vq);
      Riq.push_back(Rq);
    }

  } //end pseudopotential
  Vnl = Vplus + Vminus;
  
  //kinetic energy
  T = 0.0;
  {
    MatrixXcd Bij = walk.refHelper.Laplacian; //i = nelec , j = norbs

    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,0) * walk.refHelper.Gradient[0].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,1) * walk.refHelper.Gradient[1].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,2) * walk.refHelper.Gradient[2].row(i);
    }

    std::complex<double> DetFactor = walk.refHelper.thetaDet[0][0] * walk.refHelper.thetaDet[0][1];
    for (int i=0; i<nelec; i++) {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bij.row(i) * walk.refHelper.thetaInv[0].col(i); }
      else
      {
        if (i < walk.d.nalpha) { factor = Bij.row(i).head(walk.d.nalpha) * walk.refHelper.thetaInv[0].col(i); }
        else { factor = Bij.row(i).tail(walk.d.nbeta) * walk.refHelper.thetaInv[1].col(i - walk.d.nalpha); }
      }
      T += (DetFactor * factor).real() / DetFactor.real();
      T += walk.corrHelper.LaplaceRatio[i];
    }

    T *= -0.5;
  }
  return T + Vij + ViI + Vnl + VIJ; 
}



template<>
void rCorrelatedWavefunction<rJastrow, rMultiSlater>::updateOptVariables(Eigen::VectorXd &v) 
{
  if (schd.optimizeCps) {
    corr.updateVariables(v);
  }
  if (schd.optimizeCiCoeffs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.updateVariables(vtail);
  }
}

template<>
void rCorrelatedWavefunction<rJastrow, rMultiSlater>::getOptVariables(Eigen::VectorXd &v) const
{
  v.setZero(getNumOptVariables());
  if (schd.optimizeCps) {
    corr.getVariables(v);
  }
  if (schd.optimizeCiCoeffs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.getVariables(vtail);
  }
}

template<>
long rCorrelatedWavefunction<rJastrow, rMultiSlater>::getNumOptVariables() const
{
  long numVars = 0;
  if (schd.optimizeCps) numVars += getNumJastrowVariables();
  if (schd.optimizeCiCoeffs) numVars += getNumRefVariables();
  return numVars;
}

template<>
double rCorrelatedWavefunction<rJastrow, rMultiSlater>::HamOverlap(rWalker<rJastrow, rMultiSlater>& walk, Eigen::VectorXd& gradRatio, Eigen::VectorXd& hamRatio) const
{
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha + nbeta;

  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  gradRatio.setZero(getNumOptVariables());
  hamRatio.setZero(getNumOptVariables());

  double potentialij = 0.0, potentiali = 0.0, potentiali_ppl = 0.0, potentiali_ppnl = 0.0, potentialN = 0.0;

  //get potential
  for (int i=0; i<nelec; i++)
    for (int j=i+1; j<nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  const Pseudopotential &pp = *schd.pseudo;
  std::vector<std::vector<double>> viq; //matrix elements of nonlocal potential for every quadrature point [elec][quad]
  std::vector<std::vector<Vector3d>> riq; //coordinates corresponding to above matrix elements
  if (pp.size())
  {
    //local potential
    potentiali_ppl += pp.localPotential(walk.d);

    //nonlocal potential matrix elements and coordinates
    pp.nonLocalPotential(walk.d, viq, riq); 
  }

  Eigen::VectorXd CIgradRatio = VectorXd::Zero(0);
  Eigen::VectorXd CIhamRatio = VectorXd::Zero(0);
  if (schd.optimizeCiCoeffs)
  {
    CIgradRatio.setZero(ref.getNumOfDets());
    CIhamRatio.setZero(ref.getNumOfDets());
  } 

  //kinetic energy and nonlocal potential
  double knl = 0.0;
  {
    for (int sz = 0; sz < 2; sz++)
    {
      int shift = 0;
      if (sz == 1) { shift = nalpha; }

      //build B matrix for kinetic energy
      Eigen::MatrixXcd B = - 0.5 * walk.refHelper.Lap[sz];
      for (int i = 0; i < B.rows(); i++)
      {
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Grad[sz][0].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Grad[sz][1].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Grad[sz][2].row(i);

        B.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.A[sz].row(i);
      }
      
      //build Bbar matrix for kinetic energy
      Eigen::MatrixXcd Bbar = - 0.5 * walk.refHelper.Lapbar[sz];
      for (int i = 0; i < Bbar.rows(); i++)
      {
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Gradbar[sz][0].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Gradbar[sz][1].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Gradbar[sz][2].row(i);

        Bbar.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.Abar[sz].row(i);
      }

      //build B and Bbar matrix for pseudopotential
      if (pp.size())
      {
        for (int i = 0; i < B.rows(); i++)
        {
          for (int q = 0; q < riq[i + shift].size(); q++)
          {
            //jastrow ratio
            double Jratio = walk.corrHelper.OverlapRatio(i + shift, riq[i + shift][q], corr, walk.d);

            //atomic orbitals
            schd.basis->eval(riq[i + shift][q], &walk.refHelper.aoValues[0]);
            VectorXd ao(norbs);
            for (int j = 0; j < norbs; j++) { ao(j) = walk.refHelper.aoValues[j]; }

            //molecular orbitals
            //occ
            VectorXcd mo(ref.ref[sz].size());
            for (int j = 0; j < ref.ref[sz].size(); j++)
            {
              int orb = ref.ref[sz].at(j);
              mo(j) = ao.transpose() * ref.getHforbs(sz).col(orb);
            }
            //unocc
            VectorXcd mobar(ref.open[sz].size());
            for (int j = 0; j < ref.open[sz].size(); j++)
            {
              int orb = ref.open[sz].at(j);
              mobar(j) = ao.transpose() * ref.getHforbs(sz).col(orb);
            }

            //accumulate
            B.row(i) += viq[i + shift][q] * Jratio * mo.transpose();
            Bbar.row(i) += viq[i + shift][q] * Jratio * mobar.transpose();
          }
        }
      }

      //intermediates
      Eigen::MatrixXcd Mbar = Bbar - B * walk.refHelper.AinvAbar[sz];
      Mbar = walk.refHelper.Ainv[sz] * Mbar;
      Eigen::MatrixXcd alpha = walk.refHelper.AinvAbar[sz];

      //reference
      double refknl = 0.0;
      for (int i = 0; i < B.rows(); i++)
      {
        std::complex<double> factor = B.row(i) * walk.refHelper.Ainv[sz].col(i);
        refknl += factor.real();
      }
      knl += refknl;

      if (schd.optimizeCiCoeffs)
      {
        CIgradRatio(0) = ref.ciParity[0] * (walk.refHelper.detRatios[0][0] * walk.refHelper.detRatios[0][1] / walk.refHelper.totalRatio).real();
        CIhamRatio(0) = ref.ciParity[0] * (refknl * walk.refHelper.detRatios[0][0] * walk.refHelper.detRatios[0][1] / walk.refHelper.totalRatio).real();
      }
      
      //sum over ci wavefunctions
      for (size_t I = 1; I < ref.getNumOfDets(); I++)
      {
        Eigen::VectorXi rowVec = ref.ciIndices[sz][I][0];
        Eigen::VectorXi colVec = ref.ciIndices[sz][I][1];

        Eigen::MatrixXcd aI;
        igl::slice(alpha, rowVec, colVec, aI);
        
        Eigen::MatrixXcd MbarI;
        igl::slice(Mbar, rowVec, colVec, MbarI);

        //alpha and beta
        double cI = ref.ciParity[I] * ref.ciCoeffs[I];
        std::complex<double> ratioI = walk.refHelper.detRatios[I][0] * walk.refHelper.detRatios[I][1];
        
        Eigen::FullPivLU<MatrixXcd> lu(aI);
        if (!lu.isInvertible()) { continue; }
        Eigen::MatrixXcd aIinv = lu.inverse();
        
        knl += (cI * (aIinv * MbarI).trace() * ratioI / walk.refHelper.totalRatio).real();

        if (schd.optimizeCiCoeffs)
        {
          CIgradRatio(I) = ref.ciParity[I] * (ratioI / walk.refHelper.totalRatio).real();
          CIhamRatio(I) = ref.ciParity[I] * ((refknl + (aIinv * MbarI).trace()) * ratioI / walk.refHelper.totalRatio).real();
        }

      }
    }
  }

  double Eloc = knl + potentialij + potentiali + potentiali_ppl + potentiali_ppnl + potentialN; 

  // *********calculate gradRatio and hamRatio for jastrows
  VectorXd CPSgradRatio = VectorXd::Zero(0);
  VectorXd CPShamRatio = VectorXd::Zero(0);
  if (schd.optimizeCps)
  {
    CPSgradRatio = VectorXd::Zero(getNumJastrowVariables());
    CPShamRatio = VectorXd::Zero(getNumJastrowVariables());

    //kinetic energy part
    for (int mu = 0; mu < getNumJastrowVariables(); mu++)
    {
      CPSgradRatio(mu) = walk.corrHelper.ParamValues[mu];
      
      for (int sz = 0; sz < 2; sz++)
      {
        int shift = 0;
        if (sz == 1) { shift = nalpha; }

        //build B_mu matrix for kinetic energy
        Eigen::MatrixXcd Bmu = Eigen::MatrixXd::Zero(walk.refHelper.Lap[sz].rows(), walk.refHelper.Lap[sz].cols());
        for (int i = 0; i < Bmu.rows(); i++)
        {
          Bmu.row(i) += - walk.corrHelper.ParamGradient[0](i + shift, mu) * walk.refHelper.Grad[sz][0].row(i);
          Bmu.row(i) += - walk.corrHelper.ParamGradient[1](i + shift, mu) * walk.refHelper.Grad[sz][1].row(i);
          Bmu.row(i) += - walk.corrHelper.ParamGradient[2](i + shift, mu) * walk.refHelper.Grad[sz][2].row(i);

          Bmu.row(i) += - 0.5 * walk.corrHelper.ParamLaplacian(i + shift, mu) * walk.refHelper.A[sz].row(i);

          Bmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.corrHelper.ParamGradient[0](i + shift, mu) * walk.refHelper.A[sz].row(i);
          Bmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.corrHelper.ParamGradient[1](i + shift, mu) * walk.refHelper.A[sz].row(i);
          Bmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.corrHelper.ParamGradient[2](i + shift, mu) * walk.refHelper.A[sz].row(i);
        }
      
        //build Bbar_mu matrix for kinetic energy
        Eigen::MatrixXcd Bbarmu = Eigen::MatrixXd::Zero(walk.refHelper.Lapbar[sz].rows(), walk.refHelper.Lapbar[sz].cols());
        for (int i = 0; i < Bbarmu.rows(); i++)
        {
          Bbarmu.row(i) += - walk.corrHelper.ParamGradient[0](i + shift, mu) * walk.refHelper.Gradbar[sz][0].row(i);
          Bbarmu.row(i) += - walk.corrHelper.ParamGradient[1](i + shift, mu) * walk.refHelper.Gradbar[sz][1].row(i);
          Bbarmu.row(i) += - walk.corrHelper.ParamGradient[2](i + shift, mu) * walk.refHelper.Gradbar[sz][2].row(i);

          Bbarmu.row(i) += - 0.5 * walk.corrHelper.ParamLaplacian(i + shift, mu) * walk.refHelper.Abar[sz].row(i);

          Bbarmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.corrHelper.ParamGradient[0](i + shift, mu) * walk.refHelper.Abar[sz].row(i);
          Bbarmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.corrHelper.ParamGradient[1](i + shift, mu) * walk.refHelper.Abar[sz].row(i);
          Bbarmu.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.corrHelper.ParamGradient[2](i + shift, mu) * walk.refHelper.Abar[sz].row(i);
        }

        //intermediates
        Eigen::MatrixXcd Mbarmu = Bbarmu - Bmu * walk.refHelper.AinvAbar[sz];
        Mbarmu = walk.refHelper.Ainv[sz] * Mbarmu;
        Eigen::MatrixXcd alpha = walk.refHelper.AinvAbar[sz];

        //reference
        double refmu = 0.0;
        for (int i = 0; i < Bmu.rows(); i++)
        {
          std::complex<double> factor = Bmu.row(i) * walk.refHelper.Ainv[sz].col(i);
          refmu += factor.real();
        }
        CPShamRatio(mu) += refmu;
      
        //sum over ci wavefunctions
        for (size_t I = 1; I < ref.getNumOfDets(); I++)
        {
          Eigen::VectorXi rowVec = ref.ciIndices[sz][I][0];
          Eigen::VectorXi colVec = ref.ciIndices[sz][I][1];

          Eigen::MatrixXcd aI;
          igl::slice(alpha, rowVec, colVec, aI);
        
          Eigen::MatrixXcd MbarImu;
          igl::slice(Mbarmu, rowVec, colVec, MbarImu);

          //alpha and beta
          double cI = ref.ciParity[I] * ref.ciCoeffs[I];
          std::complex<double> ratioI = walk.refHelper.detRatios[I][0] * walk.refHelper.detRatios[I][1];
          
          Eigen::FullPivLU<MatrixXcd> lu(aI);
          if (!lu.isInvertible()) { continue; }
          Eigen::MatrixXcd aIinv = lu.inverse();
          
          CPShamRatio(mu) += (cI * (aIinv * MbarImu).trace() * ratioI / walk.refHelper.totalRatio).real();
        }
      }
    }

    //nonlocal potential energy part
    if (pp.size())
    {
      for (int i = 0; i < nelec; i++)
      {
        for (int q = 0; q < riq[i].size(); q++)
        {
          //corr ratio and param gradient
          VectorXd g = VectorXd::Zero(getNumJastrowVariables());
          double corrRatio = walk.corrHelper.OverlapRatioAndParamGradient(i, riq[i][q], corr, walk.d, g);
          double refRatio = walk.refHelper.getDetFactor(i, riq[i][q], walk.d, ref);
          //gradient element
          CPShamRatio += viq[i][q] * refRatio * corrRatio * g;
        }
      }
    }

    CPShamRatio[corr.EEsameSpinIndex] = 0.0;
    CPShamRatio[corr.EEoppositeSpinIndex] = 0.0;
    if (schd.noENCusp || schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { CPShamRatio[corr.ENIndex + I * corr.Qmax] = 0.0; }
    CPSgradRatio[corr.EEsameSpinIndex] = 0.0;
    CPSgradRatio[corr.EEoppositeSpinIndex] = 0.0;
    if (schd.noENCusp || schd.addENCusp) for (int I = 0; I < schd.uniqueAtoms.size(); I++) { CPSgradRatio[corr.ENIndex + I * corr.Qmax] = 0.0; }
    CPShamRatio += Eloc * CPSgradRatio;
  } //CPS param gradients

  hamRatio << CPShamRatio, CIhamRatio;
  gradRatio << CPSgradRatio, CIgradRatio;
  return Eloc;
}


template<>
void rCorrelatedWavefunction<rJastrow, rMultiSlater>::enforceCusp() {
}

template<>
double rCorrelatedWavefunction<rJastrow, rMultiSlater>::rHam(rWalker<rJastrow, rMultiSlater>& walk) const
{
  int norbs = Determinant::norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  
  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  //get potential
  double potentialij = 0.0, potentiali = 0.0, potentiali_ppl = 0.0, potentiali_ppnl = 0.0, potentialN = 0.0;

  for (int i=0; i<walk.d.nelec; i++)
    for (int j=i+1; j<walk.d.nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  const Pseudopotential &pp = *schd.pseudo;
  std::vector<std::vector<double>> viq; //matrix elements of nonlocal potential for every quadrature point [elec][quad]
  std::vector<std::vector<Vector3d>> riq; //coordinates corresponding to above matrix elements
  if (pp.size())
  {
    //local potential
    potentiali_ppl += pp.localPotential(walk.d);

    //nonlocal potential matrix elements and coordinates
    pp.nonLocalPotential(walk.d, viq, riq); 
  }
 
  //kinetic energy and nonlocal potential
  double knl = 0.0;
  {
    for (int sz = 0; sz < 2; sz++)
    {
      int shift = 0;
      if (sz == 1) { shift = nalpha; }

      //build B matrix for kinetic energy
      Eigen::MatrixXcd B = - 0.5 * walk.refHelper.Lap[sz];
      for (int i = 0; i < B.rows(); i++)
      {
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Grad[sz][0].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Grad[sz][1].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Grad[sz][2].row(i);

        B.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.A[sz].row(i);
      }
      
      //build Bbar matrix for kinetic energy
      Eigen::MatrixXcd Bbar = - 0.5 * walk.refHelper.Lapbar[sz];
      for (int i = 0; i < Bbar.rows(); i++)
      {
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Gradbar[sz][0].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Gradbar[sz][1].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Gradbar[sz][2].row(i);

        Bbar.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.Abar[sz].row(i);
      }

      //build B and Bbar matrix for pseudopotential
      if (pp.size())
      {
        for (int i = 0; i < B.rows(); i++)
        {
          for (int q = 0; q < riq[i + shift].size(); q++)
          {
            //jastrow ratio
            double Jratio = walk.corrHelper.OverlapRatio(i + shift, riq[i + shift][q], corr, walk.d);

            //atomic orbitals
            schd.basis->eval(riq[i + shift][q], &walk.refHelper.aoValues[0]);
            VectorXd ao(norbs);
            for (int j = 0; j < norbs; j++) { ao(j) = walk.refHelper.aoValues[j]; }

            //molecular orbitals
            //occ
            VectorXcd mo(ref.ref[sz].size());
            for (int j = 0; j < ref.ref[sz].size(); j++)
            {
              int orb = ref.ref[sz].at(j);
              mo(j) = ao.transpose() * ref.getHforbs(sz).col(orb);
            }
            //unocc
            VectorXcd mobar(ref.open[sz].size());
            for (int j = 0; j < ref.open[sz].size(); j++)
            {
              int orb = ref.open[sz].at(j);
              mobar(j) = ao.transpose() * ref.getHforbs(sz).col(orb);
            }

            //accumulate
            B.row(i) += viq[i + shift][q] * Jratio * mo.transpose();
            Bbar.row(i) += viq[i + shift][q] * Jratio * mobar.transpose();
          }
        }
      }

      //intermediates
      Eigen::MatrixXcd Mbar = Bbar - B * walk.refHelper.AinvAbar[sz];
      Mbar = walk.refHelper.Ainv[sz] * Mbar;
      Eigen::MatrixXcd alpha = walk.refHelper.AinvAbar[sz];

      //reference
      double refknl = 0.0;
      for (int i = 0; i < B.rows(); i++)
      {
          std::complex<double> factor = B.row(i) * walk.refHelper.Ainv[sz].col(i);
          refknl += factor.real();
      }
      knl += refknl;
      
      //sum over ci wavefunctions
      for (size_t I = 1; I < ref.getNumOfDets(); I++)
      {
        Eigen::VectorXi rowVec = ref.ciIndices[sz][I][0];
        Eigen::VectorXi colVec = ref.ciIndices[sz][I][1];

        Eigen::MatrixXcd aI;
        igl::slice(alpha, rowVec, colVec, aI);
        
        Eigen::MatrixXcd MbarI;
        igl::slice(Mbar, rowVec, colVec, MbarI);

        //alpha and beta
        double cI = ref.ciParity[I] * ref.ciCoeffs[I];
        std::complex<double> ratioI = walk.refHelper.detRatios[I][0] * walk.refHelper.detRatios[I][1];
        
        Eigen::FullPivLU<MatrixXcd> lu(aI);
        if (!lu.isInvertible()) { continue; }
        Eigen::MatrixXcd aIinv = lu.inverse();
        
        knl += (cI * (aIinv * MbarI).trace() * ratioI / walk.refHelper.totalRatio).real();
      }
    }
  }
  //cout << "rHam" << endl;
  //cout << knl << " " << potentialij << " " << potentiali << " " << potentiali_ppl << " " << potentiali_ppnl << " " << potentialN << endl;
  //cout << endl;
  return knl + potentialij + potentiali + potentiali_ppl + potentiali_ppnl + potentialN; 
}

//this is used only in the dmc algorithm, generates tmoves with the corresponding effective hamiltonian
//size-consistent T moves
//J. Chem. Phys. 132, 154113 (2010)
double rCorrelatedWavefunction<rJastrow, rMultiSlater>::rHam(rWalker<rJastrow, rMultiSlater>& walk, double& T, double& Vij, double& ViI, double& Vnl, double& VIJ, std::vector<std::vector<double>> &Viq, std::vector<std::vector<Vector3d>> &Riq) const
{
  int norbs = Determinant::norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  
  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  //get potential
  Vij = 0.0, ViI = 0.0, VIJ = 0.0, Vnl = 0.0;

  for (int i=0; i<walk.d.nelec; i++)
    for (int j=i+1; j<walk.d.nelec; j++) {
      Vij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      ViI -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      VIJ += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  const Pseudopotential &pp = *schd.pseudo;
  double Vminus = 0.0, Vplus = 0.0;
  if (pp.size())
  {
    //local potential
    ViI += pp.localPotential(walk.d);

    //nonlocal potential
    std::vector<std::vector<double>> viq; //matrix elements of nonlocal potential for every quadrature point [elec][quad]
    std::vector<std::vector<Vector3d>> riq; //coordinates corresponding to above matrix elements
    pp.nonLocalPotential(walk.d, viq, riq); 

    //calculate wavefunction overlap ratios with riq;
    std::vector<std::vector<double>> Yiq;
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> Yq;
      for (int q = 0; q < riq[i].size(); q++)
      {
        double ovlpRatio = getOverlapFactor(i, riq[i][q], walk);
        Yq.push_back(ovlpRatio);
      }
      Yiq.push_back(Yq);
    }

    //calculate nonlocal potential and matrix elements for t moves
    Viq.clear();
    Riq.clear();
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> Vq;
      std::vector<Vector3d> Rq;
      for (int q = 0; q < riq[i].size(); q++)
      {
        //matrix element
        double vxpx = viq[i][q] * Yiq[i][q];

        if (vxpx < 0.0)
        {
          Vminus += vxpx;
          Vq.push_back(vxpx);
          Rq.push_back(riq[i][q]);
        }
        else
        {
          Vplus += vxpx;
        }
      }
      //update tensors
      Viq.push_back(Vq);
      Riq.push_back(Rq);
    }

  } //end pseudopotential
  Vnl = Vplus + Vminus;
 
  //kinetic energy
  T = 0.0;
  {
    for (int sz = 0; sz < 2; sz++)
    {
      int shift = 0;
      if (sz == 1) { shift = nalpha; }

      //build B matrix for kinetic energy
      Eigen::MatrixXcd B = - 0.5 * walk.refHelper.Lap[sz];
      for (int i = 0; i < B.rows(); i++)
      {
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Grad[sz][0].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Grad[sz][1].row(i);
        B.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Grad[sz][2].row(i);

        B.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.A[sz].row(i);
      }
      
      //build Bbar matrix for kinetic energy
      Eigen::MatrixXcd Bbar = - 0.5 * walk.refHelper.Lapbar[sz];
      for (int i = 0; i < Bbar.rows(); i++)
      {
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 0) * walk.refHelper.Gradbar[sz][0].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 1) * walk.refHelper.Gradbar[sz][1].row(i);
        Bbar.row(i) += - walk.corrHelper.GradRatio(i + shift, 2) * walk.refHelper.Gradbar[sz][2].row(i);

        Bbar.row(i) += - 0.5 * walk.corrHelper.LaplaceRatio(i + shift) * walk.refHelper.Abar[sz].row(i);
      }

      //intermediates
      Eigen::MatrixXcd Mbar = Bbar - B * walk.refHelper.AinvAbar[sz];
      Mbar = walk.refHelper.Ainv[sz] * Mbar;
      Eigen::MatrixXcd alpha = walk.refHelper.AinvAbar[sz];

      //reference
      double refT = 0.0;
      for (int i = 0; i < B.rows(); i++)
      {
          std::complex<double> factor = B.row(i) * walk.refHelper.Ainv[sz].col(i);
          refT += factor.real();
      }
      T += refT;
      
      //sum over ci wavefunctions
      for (size_t I = 1; I < ref.getNumOfDets(); I++)
      {
        Eigen::VectorXi rowVec = ref.ciIndices[sz][I][0];
        Eigen::VectorXi colVec = ref.ciIndices[sz][I][1];

        Eigen::MatrixXcd aI;
        igl::slice(alpha, rowVec, colVec, aI);
        
        Eigen::MatrixXcd MbarI;
        igl::slice(Mbar, rowVec, colVec, MbarI);

        //alpha and beta
        double cI = ref.ciParity[I] * ref.ciCoeffs[I];
        std::complex<double> ratioI = walk.refHelper.detRatios[I][0] * walk.refHelper.detRatios[I][1];
        
        Eigen::FullPivLU<MatrixXcd> lu(aI);
        if (!lu.isInvertible()) { continue; }
        Eigen::MatrixXcd aIinv = lu.inverse();
        
        T += (cI * (aIinv * MbarI).trace() * ratioI / walk.refHelper.totalRatio).real();
      }
    }
  }
  return T + Vij + ViI + Vnl + VIJ; 
}



template<>
void rCorrelatedWavefunction<rJastrow, rBFSlater>::updateOptVariables(Eigen::VectorXd &v) 
{
  if (schd.optimizeCps) {
    corr.updateVariables(v);
  }
  if (schd.optimizeOrbs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.updateVariables(vtail);
  }
}

template<>
void rCorrelatedWavefunction<rJastrow, rBFSlater>::getOptVariables(Eigen::VectorXd &v) const
{
  v.setZero(getNumOptVariables());
  if (schd.optimizeCps) {
    corr.getVariables(v);
  }
  if (schd.optimizeOrbs) {
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.getVariables(vtail);
  }
}

template<>
long rCorrelatedWavefunction<rJastrow, rBFSlater>::getNumOptVariables() const
{
  long numVars = 0;
  if (schd.optimizeCps) numVars += getNumJastrowVariables();
  if (schd.optimizeOrbs) numVars += getNumRefVariables();
  return numVars;
}

template<>
double rCorrelatedWavefunction<rJastrow, rBFSlater>::rHam(rWalker<rJastrow, rBFSlater>& walk) const {
  int norbs = Determinant::norbs;

  //init jastrow gradient and laplacian
  walk.corrHelper.GradientAndLaplacian(walk.d);

  double potentialij = 0.0, potentiali = 0.0, potentiali_pp = 0.0, potentialN = 0.0;

  //get potential
  for (int i=0; i<walk.d.nelec; i++)
    for (int j=i+1; j<walk.d.nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }

  for (int i=0; i<walk.d.nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
    }
  }

  //pseudopotential
  const Pseudopotential &pp = *schd.pseudo;
  if (pp.size() != 0) //if pseudopotential object is not empty
  {
    //local potential
    potentiali_pp += pp.localPotential(walk.d);

    //nonlocal potential
    std::vector<std::vector<double>> viq;
    std::vector<std::vector<Vector3d>> riq;
    pp.nonLocalPotential(walk.d, viq, riq);
    
    for (int i = 0; i < walk.d.nelec; i++)
    {
      for (int q = 0; q < riq[i].size(); q++)
      {
        potentiali_pp += viq[i][q] * getOverlapFactor(i, riq[i][q], walk);
      }
    }
  }
  
  double kinetic = 0.0;  
  {
    walk.refHelper.calcSlaterDerivatives(ref, walk.d);
    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
      std::complex<double> factor = 0.0;
      factor += 2.* walk.corrHelper.GradRatio(i,0) * walk.refHelper.slaterGradientRatio[0](i);
      factor += 2.* walk.corrHelper.GradRatio(i,1) * walk.refHelper.slaterGradientRatio[1](i);
      factor += 2.* walk.corrHelper.GradRatio(i,2) * walk.refHelper.slaterGradientRatio[2](i);
      factor += walk.refHelper.slaterLaplacianRatio(i);
      kinetic += (walk.refHelper.thetaDet * factor).real() / walk.refHelper.thetaDet.real();
      kinetic += walk.corrHelper.LaplaceRatio[i];
    }
  }
  //cout << -0.5*(kinetic) << " " << potentialij << " " << potentiali << " " << potentiali_pp << " " << potentialN << endl;
  return -0.5*(kinetic) + potentialij + potentiali + potentiali_pp + potentialN; 
}

template<>
double rCorrelatedWavefunction<rJastrow, rBFSlater>::HamOverlap(rWalker<rJastrow, rBFSlater>& walk,
                                                              Eigen::VectorXd& gradRatio,
                                                              Eigen::VectorXd& hamRatio) const
{
  return 0.;
}

template<>
void rCorrelatedWavefunction<rJastrow, rBFSlater>::enforceCusp() 
{

}
