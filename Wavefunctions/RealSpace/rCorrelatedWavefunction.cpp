#include "rCorrelatedWavefunction.h"
#include "rPseudopotential.h"
#include "global.h"
#include "input.h"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

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
  MatrixXcd Bnl;        //Bnl(elec, mo) = Vnl * DetMatrix     
  MatrixXd AOBnl;        //AOBnl(elec, ao) = \partial_{mocoeff} Vnl * DetMatrix    
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
    {
      const ppHelper &ppatm = it->second;
      for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
      {
        int I = ppatm.indices()[a];
        auto it1 = ppatm.begin();
        int l = it1->first;
        if (l == -1) //first angular momentum channel should be local potential
        {
          const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
          for (int i = 0; i < nelec; i++) //loop over electrons
          {
             //calculate potential
             double v = 0.0;
             for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(walk.RiN(i, I), pec[m] - 2)  * std::exp(-pec[m + 1] * walk.RiN(i, I) * walk.RiN(i, I)) * pec[m + 2]; }
             potentiali_ppl += v;    
          }
        }
      }
    }

    //nonlocal potential
    int nao = schd.hf == "ghf" ? 2*norbs : norbs; //num atomic orbitals
    
    Bnl = MatrixXd::Zero(nelec, nelec);
    AOBnl = MatrixXd::Zero(nelec, nao);
    for (int i = 0; i < nelec; i++) //loop over electrons
    {
      for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
      {
        const ppHelper &ppatm = it->second;
        for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
        {
          int I = ppatm.indices()[a];
          for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
          {
            int l = it1->first; //angular momentum
            const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
      
            Vector3d rI = schd.Ncoords[I];
            Vector3d ri = walk.d.coord[i];
            Vector3d riI = ri - rI;

            //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
            if (l == -1 || riI.norm() > schd.pCutOff) { continue; } 

            //calculate potential
            double v = 0.0;
            for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2)  * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }

            //random rotation
            Vector3d riIhat = riI.normalized();
            double angle = walk.uR() * 2.0 * M_PI;
            //double angle = 0.0; //this is easier when debugging
            AngleAxis<double> rot(angle, riIhat);

            VectorXd Integral = VectorXd::Zero(norbs);
            double C = (2.0 * double(l) + 1.0) / (4.0 * M_PI);
            for (int q = 0; q < schd.Q.size(); q++)
            {
              //calculate new vector, riprime
              Vector3d riIprime = riI.norm() * (rot * schd.Q[q]);
              Vector3d riprime = riIprime + rI;

              //eval basis
              schd.basis->eval(riprime, &walk.refHelper.aoValues[0]);

              //calculate angle
              double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());

              //multiply legendre polynomial and jastrow overlap ratio
              double f = boost::math::legendre_p<double>(l, costheta) * walk.corrHelper.OverlapRatio(i, riprime, corr, walk.d);

              for (int j = 0; j < norbs; j++)
              {
                Integral(j) += C * f * walk.refHelper.aoValues[j] * 4.0 * M_PI / double(schd.Q.size());
              }
            }

            //update AOBnl
            if (i < nalpha)
            {
              AOBnl.row(i).head(norbs) += v * Integral;
            }
            else
            {
              AOBnl.row(i).tail(norbs) += v * Integral;
            }  

          }
        }
      }
    }

    for (int i = 0; i < nelec; i++)
    {
      //options for rhf/uhf
      int sz = i < nalpha ? 0 : 1;
      int nmo = i < nalpha ? nalpha : nbeta;

      if (schd.hf == "ghf")
      {
        for (int mo = 0; mo < nelec; mo++) 
        {
          for (int j = 0; j < norbs; j++)
          {
            int J = i < nalpha ? j : j + norbs;
            Bnl(i, mo) += AOBnl(i, J) * ref.getHforbs(0)(J, mo);
          }
        }
      }
      else //rhf/ufh
      {
        for (int mo = 0; mo < nmo; mo++)
        {
          for (int j = 0; j < norbs; j++)
          {
            Bnl(i, mo + sz * nalpha) += AOBnl(i, j) * ref.getHforbs(sz)(j, mo);
          }
        }
      }
    }

    for (int i = 0; i < nelec; i++)
    {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bnl.row(i) * thetaInv[0].col(i); }
      else
      {
        if (i < walk.d.nalpha) { factor = Bnl.row(i).head(walk.d.nalpha) * thetaInv[0].col(i); }
        else { factor = Bnl.row(i).tail(walk.d.nbeta) * thetaInv[1].col(i - walk.d.nalpha); }
      }
      potentiali_ppnl += (thetaDet * factor).real() / thetaDet.real();
      cpotentiali_ppnl += factor;
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
  //cout << "T\t\tVij\t\tViI\t\tVl\t\tVnl\t\tVIJ" << endl;
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
      //VectorXcd Bij = VectorXcd::Zero(nelec);
      for (int j = 0; j < corr._params.size(); j++)
      {
          CPSgradRatio[j] = walk.corrHelper.ParamValues[j];
          //CPShamRatio[j] = 0.0;
          VectorXcd Bij;
          for (int i = 0; i < nelec; i++)
          {
              Bij  = -walk.corrHelper.ParamGradient[0](i,j) * walk.refHelper.Gradient[0].row(i);
              Bij += -walk.corrHelper.ParamGradient[1](i,j) * walk.refHelper.Gradient[1].row(i);
              Bij += -walk.corrHelper.ParamGradient[2](i,j) * walk.refHelper.Gradient[2].row(i);
        
              //std::complex<double> factor = Bij.transpose() * thetaInv.col(i);
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
        for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
        {
          const ppHelper &ppatm = it->second;
          for (int j = 0; j < ppatm.indices().size(); j++) //loop over indices of atom
          {
            int I = ppatm.indices()[j];
            for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
            {
              int l = it1->first; //angular momentum
              const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
              for (int i = 0; i < walk.d.nelec; i++) //loop over electrons
              {
                Vector3d rI = schd.Ncoords[I];
                Vector3d ri = walk.d.coord[i];
                Vector3d riI = ri - rI;
                
                //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
                if (l == -1 || riI.norm() > schd.pCutOff) { continue; } 

                //calculate potential
                double v = 0.0;
                for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2)  * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }

                //random rotation
                Vector3d riIhat = riI.normalized();
                double angle = walk.uR() * 2.0 * M_PI;
                //double angle = 0.0; //this is easier when debugging
                AngleAxis<double> rot(angle, riIhat);

                VectorXd G = VectorXd::Zero(CPShamRatio.size());
                double C = (2.0 * double(l) + 1.0) / (4.0 * M_PI);
                for (int q = 0; q < schd.Q.size(); q++)
                {
                  //calculate new vector, riprime
                  Vector3d riIprime = riI.norm() * (rot * schd.Q[q]);
                  Vector3d riprime = riIprime + rI;

                  //calculate angle
                  double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());

                  //multiply legendre polynomial and jastrow overlap ratio
                  VectorXd g;
                  double ratio = walk.refHelper.getDetFactor(i, riprime, walk.d, ref) * walk.corrHelper.OverlapRatioAndParamGradient(i, riprime, corr, walk.d, g);
                  G += C * boost::math::legendre_p<double>(l, costheta) * ratio * g * 4.0 * M_PI / double(schd.Q.size());
                }

                //update CPShamRatio
                CPShamRatio += v * G;
              }
            }
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
  MatrixXcd Bnl;        //Bnl(elec, mo) = Vnl * DetMatrix     
  MatrixXd AOBnl;        //AOBnl(elec, ao) = \partial_{mocoeff} Vnl * DetMatrix    
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
    {
      const ppHelper &ppatm = it->second;
      for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
      {
        int I = ppatm.indices()[a];
        auto it1 = ppatm.begin();
        int l = it1->first;
        if (l == -1) //first angular momentum channel should be local potential
        {
          const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
          for (int i = 0; i < nelec; i++) //loop over electrons
          {
             //calculate potential
             double v = 0.0;
             for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(walk.RiN(i, I), pec[m] - 2)  * std::exp(-pec[m + 1] * walk.RiN(i, I) * walk.RiN(i, I)) * pec[m + 2]; }
             potentiali_pp += v;    
          }
        }
      }
    }

    //nonlocal potential
    int nmo = nelec; //ghf num molecular orbitals
    int nao = schd.hf == "ghf" ? 2*norbs : norbs; //num atomic orbitals
    
    Bnl = MatrixXd::Zero(nelec, nelec);
    AOBnl = MatrixXd::Zero(nelec, nao);
    for (int i = 0; i < nelec; i++) //loop over electrons
    {
      for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
      {
        const ppHelper &ppatm = it->second;
        for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
        {
          int I = ppatm.indices()[a];
          for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
          {
            int l = it1->first; //angular momentum
            const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
      
            Vector3d rI = schd.Ncoords[I];
            Vector3d ri = walk.d.coord[i];
            Vector3d riI = ri - rI;

            //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
            if (l == -1 || riI.norm() > schd.pCutOff) { continue; } 

            //calculate potential
            double v = 0.0;
            for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2)  * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }

            //random rotation
            Vector3d riIhat = riI.normalized();
            double angle = walk.uR() * 2.0 * M_PI;
            //double angle = 0.0; //this is easier when debugging
            AngleAxis<double> rot(angle, riIhat);

            VectorXd Integral = VectorXd::Zero(norbs);
            double C = (2.0 * double(l) + 1.0) / (4.0 * M_PI);
            for (int q = 0; q < schd.Q.size(); q++)
            {
              //calculate new vector, riprime
              Vector3d riIprime = riI.norm() * (rot * schd.Q[q]);
              Vector3d riprime = riIprime + rI;

              //eval basis
              schd.basis->eval(riprime, &walk.refHelper.aoValues[0]);

              //calculate angle
              double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());

              //multiply legendre polynomial and jastrow overlap ratio
              double f = boost::math::legendre_p<double>(l, costheta) * walk.corrHelper.OverlapRatio(i, riprime, corr, walk.d);

              for (int j = 0; j < norbs; j++)
              {
                Integral(j) += C * f * walk.refHelper.aoValues[j] * 4.0 * M_PI / double(schd.Q.size());
              }
            }

            //update AOBnl
            if (i < nalpha)
            {
              AOBnl.row(i).head(norbs) += v * Integral;
            }
            else
            {
              AOBnl.row(i).tail(norbs) += v * Integral;
            }  

          }
        }
      }
    }

    for (int i = 0; i < nelec; i++)
    {
      //options for rhf/uhf
      int sz = i < nalpha ? 0 : 1;
      int nmo = i < nalpha ? nalpha : nbeta;

      if (schd.hf == "ghf")
      {
        for (int mo = 0; mo < nelec; mo++) 
        {
          for (int j = 0; j < norbs; j++)
          {
            int J = i < nalpha ? j : j + norbs;
            Bnl(i, mo) += AOBnl(i, J) * ref.getHforbs(0)(J, mo);
          }
        }
      }
      else //rhf/ufh
      {
        for (int mo = 0; mo < nmo; mo++)
        {
          for (int j = 0; j < norbs; j++)
          {
            Bnl(i, mo + sz * nalpha) += AOBnl(i, j) * ref.getHforbs(sz)(j, mo);
          }
        }
      }
    }

    std::complex<double> DetFactor = walk.refHelper.thetaDet[0][0] * walk.refHelper.thetaDet[0][1];
    for (int i = 0; i < nelec; i++)
    {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bnl.row(i) * walk.refHelper.thetaInv[0].col(i); }
      else
      {
        if (i < walk.d.nalpha) { factor = Bnl.row(i).head(walk.d.nalpha) * walk.refHelper.thetaInv[0].col(i); }
        else { factor = Bnl.row(i).tail(walk.d.nbeta) * walk.refHelper.thetaInv[1].col(i - walk.d.nalpha); }
      }
      potentiali_pp += (DetFactor * factor).real() / DetFactor.real();
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
  //cout << -0.5*(kinetic) << " " << potentialij << " " << potentiali << " " << potentiali_pp << " " << potentialN << endl;
  return -0.5*(kinetic) + potentialij + potentiali + potentiali_pp + potentialN; 
}

double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(rWalker<rJastrow, rSlater>& walk, double& T, double& Vij, double& ViI, double& Vpp, double& VIJ) const {
  int norbs = Determinant::norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

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
  MatrixXcd Bnl;        //Bnl(elec, mo) = Vnl * DetMatrix     
  MatrixXd AOBnl;        //AOBnl(elec, ao) = \partial_{mocoeff} Vnl * DetMatrix    
  if (pp.size()) //if pseudopotential object is not empty
  {
    //local potential
    for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
    {
      const ppHelper &ppatm = it->second;
      for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
      {
        int I = ppatm.indices()[a];
        auto it1 = ppatm.begin();
        int l = it1->first;
        if (l == -1) //first angular momentum channel should be local potential
        {
          const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
          for (int i = 0; i < nelec; i++) //loop over electrons
          {
             //calculate potential
             double v = 0.0;
             for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(walk.RiN(i, I), pec[m] - 2)  * std::exp(-pec[m + 1] * walk.RiN(i, I) * walk.RiN(i, I)) * pec[m + 2]; }
             potentiali_pp += v;    
          }
        }
      }
    }

    //nonlocal potential
    int nmo = nelec; //ghf num molecular orbitals
    int nao = schd.hf == "ghf" ? 2*norbs : norbs; //num atomic orbitals
    
    Bnl = MatrixXd::Zero(nelec, nelec);
    AOBnl = MatrixXd::Zero(nelec, nao);
    for (int i = 0; i < nelec; i++) //loop over electrons
    {
      for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
      {
        const ppHelper &ppatm = it->second;
        for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
        {
          int I = ppatm.indices()[a];
          for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
          {
            int l = it1->first; //angular momentum
            const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
      
            Vector3d rI = schd.Ncoords[I];
            Vector3d ri = walk.d.coord[i];
            Vector3d riI = ri - rI;

            //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
            if (l == -1 || riI.norm() > schd.pCutOff) { continue; } 

            //calculate potential
            double v = 0.0;
            for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2)  * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }

            //random rotation
            Vector3d riIhat = riI.normalized();
            double angle = walk.uR() * 2.0 * M_PI;
            //double angle = 0.0; //this is easier when debugging
            AngleAxis<double> rot(angle, riIhat);

            VectorXd Integral = VectorXd::Zero(norbs);
            double C = (2.0 * double(l) + 1.0) / (4.0 * M_PI);
            for (int q = 0; q < schd.Q.size(); q++)
            {
              //calculate new vector, riprime
              Vector3d riIprime = riI.norm() * (rot * schd.Q[q]);
              Vector3d riprime = riIprime + rI;

              //eval basis
              schd.basis->eval(riprime, &walk.refHelper.aoValues[0]);

              //calculate angle
              double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());

              //multiply legendre polynomial and jastrow overlap ratio
              double f = boost::math::legendre_p<double>(l, costheta) * walk.corrHelper.OverlapRatio(i, riprime, corr, walk.d);

              for (int j = 0; j < norbs; j++)
              {
                Integral(j) += C * f * walk.refHelper.aoValues[j] * 4.0 * M_PI / double(schd.Q.size());
              }
            }

            //update AOBnl
            if (i < nalpha)
            {
              AOBnl.row(i).head(norbs) += v * Integral;
            }
            else
            {
              AOBnl.row(i).tail(norbs) += v * Integral;
            }  

          }
        }
      }
    }

    for (int i = 0; i < nelec; i++)
    {
      //options for rhf/uhf
      int sz = i < nalpha ? 0 : 1;
      int nmo = i < nalpha ? nalpha : nbeta;

      if (schd.hf == "ghf")
      {
        for (int mo = 0; mo < nelec; mo++) 
        {
          for (int j = 0; j < norbs; j++)
          {
            int J = i < nalpha ? j : j + norbs;
            Bnl(i, mo) += AOBnl(i, J) * ref.getHforbs(0)(J, mo);
          }
        }
      }
      else //rhf/ufh
      {
        for (int mo = 0; mo < nmo; mo++)
        {
          for (int j = 0; j < norbs; j++)
          {
            Bnl(i, mo + sz * nalpha) += AOBnl(i, j) * ref.getHforbs(sz)(j, mo);
          }
        }
      }
    }

    std::complex<double> DetFactor = walk.refHelper.thetaDet[0][0] * walk.refHelper.thetaDet[0][1];
    for (int i = 0; i < nelec; i++)
    {
      std::complex<double> factor = 0.0;
      if (schd.hf == "ghf") { factor = Bnl.row(i) * walk.refHelper.thetaInv[0].col(i); }
      else
      {
        if (i < walk.d.nalpha) { factor = Bnl.row(i).head(walk.d.nalpha) * walk.refHelper.thetaInv[0].col(i); }
        else { factor = Bnl.row(i).tail(walk.d.nbeta) * walk.refHelper.thetaInv[1].col(i - walk.d.nalpha); }
      }
      potentiali_pp += (DetFactor * factor).real() / DetFactor.real();
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
  //cout << -0.5*(kinetic) << " " << potentialij << " " << potentiali << " " << potentiali_pp << " " << potentialN << endl;
  T = -0.5 * kinetic;
  Vij = potentialij;
  ViI = potentiali;
  Vpp = potentiali_pp;
  VIJ = potentialN;
  return -0.5*(kinetic) + potentialij + potentiali + potentiali_pp + potentialN; 
}

template<>
double rCorrelatedWavefunction<rJastrow, rBFSlater>::rHam(rWalker<rJastrow, rBFSlater>& walk) const {
  int norbs = Determinant::norbs;

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
    auto random = std::bind(std::normal_distribution<double>(0.0, 1.0), std::ref(generator));
    for (auto it = pp.begin(); it != pp.end(); ++it) //loop over atoms with pseudopotential
    {
      const ppHelper &ppatm = it->second;
      for (int j = 0; j < ppatm.indices().size(); j++) //loop over indices of atom
      {
        int I = ppatm.indices()[j];
        for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
        {
          int l = it1->first; //angular momentum
          const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
          for (int i = 0; i < walk.d.nelec; i++) //loop over electrons
          {
            double Int = 1.0;
            double C = 1.0;
            Vector3d rI = schd.Ncoords[I];
            Vector3d ri = walk.d.coord[i];
            Vector3d riI = ri - rI;

            //random rotation
            Vector3d riIhat = riI.normalized();
            double angle = walk.uR() * 2.0 * M_PI;
            //double angle = 0.0; //this is easier when debugging
            AngleAxis<double> rot(angle, riIhat);
            
            //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
            if (l != -1 && riI.norm() > 2.0) { continue; } 

            //calculate potential
            double val = 0.0;
            for (int m = 0; m < pec.size(); m = m + 3) { val += std::pow(riI.norm(), pec[m] - 2)  * std::exp(-pec[m + 1] * riI.norm() * riI.norm()) * pec[m + 2]; }
            
            //integrate if nonlocal potential
            if (l != -1) //angular momentum projector
            {
              Int = 0.0;
              C = std::sqrt((2.0 * (double) l + 1.0) / (4.0 * M_PI));

              for (int v = 0; v < schd.Q.size(); v++)
              {
                  //calculate new vector, riprime
                  Vector3d riIprime = riI.norm() * (rot * schd.Q[v]);
                  //calculate angle
                  double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());
                  //multiply legendre polynomial and wavefunction overlap ratio
                  Vector3d riprime = riIprime + rI;
                  Int += boost::math::legendre_p<double>(l, costheta) * getOverlapFactor(i, riprime, walk);
              }
              Int /= (double) schd.Q.size();
              Int *= (C * 4.0 * M_PI);

            }

            potentiali_pp += val * C * Int;
          }
        }
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
