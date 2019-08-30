#include "rCorrelatedWavefunction.h"
#include "global.h"
#include "input.h"

template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::HamOverlap(const rWalker<rJastrow, rSlater>& walk,
                                                              Eigen::VectorXd& gradRatio,
                                                              Eigen::VectorXd& hamRatio) const
{
  gradRatio.setZero(getNumVariables());
  hamRatio.setZero(getNumVariables());
  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

  std::complex<double> thetaDet = walk.refHelper.thetaDet[0][0];
  MatrixXcd thetaInv = walk.refHelper.thetaInv[0];
  std::complex<double> i(0.0, 1.0);

  //true and complex local energy
  double Eloc;
  std::complex<double> cEloc;

  double potentialij = 0.0, potentiali = 0.0, potentialN = 0.0;
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
      std::complex<double> factor = Bij.row(i) * thetaInv.col(i);
      kinetic += (thetaDet * factor).real() / thetaDet.real();
      kinetic += walk.corrHelper.LaplaceRatio[i];
      ckinetic += factor;
      ckinetic += walk.corrHelper.LaplaceRatio[i];
    }
  }
  Eloc = -0.5 * kinetic + potentialij + potentiali + potentialN; 
  cEloc = -0.5 * ckinetic + potentialij + potentiali + potentialN; 
 
  int numVars = 0;
  //*********calculate gradRatio and hamRatio for jastrows
  VectorXd CPSgradRatio = VectorXd::Zero(getNumJastrowVariables());
  VectorXd CPShamRatio = VectorXd::Zero(getNumJastrowVariables());
  if (schd.optimizeCps)
  {
      numVars += corr._params.size();
      VectorXcd Bij = VectorXcd::Zero(nelec);
      for (int j = 0; j < corr._params.size(); j++)
      {
          CPSgradRatio[j] = walk.corrHelper.ParamValues[j];
          //CPShamRatio[j] = 0.0;
          for (int i = 0; i < nelec; i++)
          {
              Bij  = -walk.corrHelper.ParamGradient[0](i,j) * walk.refHelper.Gradient[0].row(i);
              Bij += -walk.corrHelper.ParamGradient[1](i,j) * walk.refHelper.Gradient[1].row(i);
              Bij += -walk.corrHelper.ParamGradient[2](i,j) * walk.refHelper.Gradient[2].row(i);
        
              std::complex<double> factor = Bij.transpose() * thetaInv.col(i);
              CPShamRatio[j] += (thetaDet * factor).real() / thetaDet.real();
              CPShamRatio[j] += -0.5*(walk.corrHelper.ParamLaplacian(i, j) +
                               2.*walk.corrHelper.GradRatio(i,0)*walk.corrHelper.ParamGradient[0](i,j)+
                               2.*walk.corrHelper.GradRatio(i,1)*walk.corrHelper.ParamGradient[1](i,j)+
                               2.*walk.corrHelper.GradRatio(i,2)*walk.corrHelper.ParamGradient[2](i,j));
          }
      }
      CPShamRatio[corr.EEsameSpinIndex] = 0.0;
      CPShamRatio[corr.EEoppositeSpinIndex] = 0.0;
      CPSgradRatio[corr.EEsameSpinIndex] = 0.0;
      CPSgradRatio[corr.EEoppositeSpinIndex] = 0.0;
      CPShamRatio += Eloc * CPSgradRatio;
  }
  
  
  //*********calculate the hamoverlap for orbitals
  VectorXd RefgradRatio = VectorXd::Zero(getNumVariables() - getNumJastrowVariables());
  VectorXd RefhamRatio = VectorXd::Zero(getNumVariables() - getNumJastrowVariables());
  VectorXcd RefGradcOvlp = VectorXcd::Zero(getNumVariables() - getNumJastrowVariables());
  VectorXcd RefGradcEloc = VectorXcd::Zero(getNumVariables() - getNumJastrowVariables());

  if (schd.optimizeOrbs) {
    
    MatrixXd AoRi = MatrixXd::Zero(nelec, 2*norbs);
    vector<double>& aoValues = const_cast<vector<double>&>(walk.refHelper.aoValues);
    aoValues.resize(norbs);  
    for (int elec=0; elec<nelec; elec++) {
      schd.basis->eval(walk.d.coord[elec], &aoValues[0]);
      for (int orb = 0; orb<norbs; orb++) 
        if (elec < nalpha)
          AoRi(elec, orb) = aoValues[orb];
        else
          AoRi(elec, norbs+orb) = aoValues[orb];
    } 
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
    
    //precalculate A-1 B A-1
    MatrixXcd X = thetaInv * Laplacian * thetaInv;
    MatrixXcd Xgx = thetaInv * Gradx * thetaInv;
    MatrixXcd Xgy = thetaInv * Grady * thetaInv;
    MatrixXcd Xgz = thetaInv * Gradz * thetaInv;

    for (int mo = 0; mo < nelec; mo++) {
      
      for (int orb = 0; orb < 2*norbs; orb++) {
        //laplacian contribution
        {
          std::complex<double> t1 = thetaInv.row(mo) * AOLaplacian.col(orb);
          std::complex<double> t2 = X.row(mo) * AoRi.col(orb);
          std::complex<double> factor = -0.5 * (t1 - t2);
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo] = factor;
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] = i * factor;
        }

        //grad contribution
        {
          std::complex<double> t1 = thetaInv.row(mo) * AOGradx.col(orb);
          std::complex<double> t2 = Xgx.row(mo) * AoRi.col(orb);
          std::complex<double> factor = -(t1 - t2);
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
        }
        {
          std::complex<double> t1 = thetaInv.row(mo) * AOGrady.col(orb);
          std::complex<double> t2 = Xgy.row(mo) * AoRi.col(orb);
          std::complex<double> factor = -(t1 - t2);
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
        }
        {
          std::complex<double> t1 = thetaInv.row(mo) * AOGradz.col(orb);
          std::complex<double> t2 = Xgz.row(mo) * AoRi.col(orb);
          std::complex<double> factor = -(t1 - t2);
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo] += factor;
          RefGradcEloc[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
        }

        std::complex<double> factor = thetaInv.row(mo) * AoRi.col(orb);
        RefGradcOvlp[numDets + 2*orb * nelec + 2*mo] += factor;
        RefGradcOvlp[numDets + 2*orb * nelec + 2*mo + 1] += i * factor;
        RefgradRatio[numDets + 2*orb * nelec + 2*mo] = (factor * thetaDet).real() / thetaDet.real();        
        if (schd.ifComplex) RefgradRatio[numDets + 2*orb * nelec + 2*mo + 1] = (i * factor * thetaDet).real() / thetaDet.real();
      } 
    }
  }
  RefhamRatio = ((cEloc * RefGradcOvlp + RefGradcEloc) * thetaDet).real() / thetaDet.real();
  hamRatio << CPShamRatio, RefhamRatio;
  gradRatio << CPSgradRatio, RefgradRatio;
  return Eloc;
}

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::enforceCusp() {
  return;
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
double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(const rWalker<rJastrow, rSlater>& walk) const {
  int norbs = Determinant::norbs;

  double potentialij = 0.0, potentiali=0;

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

  double potentialN = .0;
  for (int i=0; i<schd.Ncoords.size(); i++) {
    for (int j=i+1; j<schd.Ncoords.size(); j++) {
      potentialN += schd.Ncharge[i] * schd.Ncharge[j]/walk.RNM(i,j);
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
    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
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
  //cout << kinetic<<"  "<<potentialij<<"  "<<potentiali<<endl;
  return -0.5*(kinetic) + potentialij + potentiali + potentialN;
  
}

