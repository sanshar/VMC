#include "rCorrelatedWavefunction.h"

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::HamOverlap(const rWalker<rJastrow, rSlater>& walk,
                                                           Eigen::VectorXd& hamgrad) const {  


  int norbs = schd.basis->getNorbs();
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  int numDets = ref.determinants.size();

  MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs
  for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
    Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,0) * walk.refHelper.Gradient[0].row(i);
    Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,1) * walk.refHelper.Gradient[1].row(i);
    Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,2) * walk.refHelper.Gradient[2].row(i);
  }
  
  double kineticTot = 0.0;
  for (int i=0; i<nelec; i++) {
    kineticTot += walk.refHelper.thetaInv[0].row(i).dot(Bij.col(i));
    kineticTot += walk.corrHelper.LaplaceRatio[i];
  }

  double potentialij = 0.0, potentiali=0;

  //get potential
  for (int i=0; i<nelec; i++)
    for (int j=i+1; j<nelec; j++) {
      potentialij += 1./walk.Rij(i,j);
    }
  for (int i=0; i<nelec; i++) {
    for (int j=0; j<schd.Ncoords.size(); j++) {
      potentiali -= schd.Ncharge[j]/walk.RiN(i,j);
    }
  }

  int numVars = 0;
  //*********calculate the hamoverlap for jastrows
  if (schd.optimizeCps) {
    numVars += corr._params.size();
    for (int j=0; j<corr._params.size(); j++) {
      hamgrad[j] = 0.0;
      if (corr._jastrow.fixed[j] == 1) continue;
      for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
        Bij.row(i)  = -walk.corrHelper.ParamGradient[j](i,0) * walk.refHelper.Gradient[0].row(i);
        Bij.row(i) += -walk.corrHelper.ParamGradient[j](i,1) * walk.refHelper.Gradient[1].row(i);
        Bij.row(i) += -walk.corrHelper.ParamGradient[j](i,2) * walk.refHelper.Gradient[2].row(i);

        hamgrad[j] += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
        hamgrad[j] += -0.5* (walk.corrHelper.ParamLaplacianIntermediate(i, j) +
                          2.*walk.corrHelper.GradRatio(i,0)*walk.corrHelper.ParamGradient[j](i,0)+
                          2.*walk.corrHelper.GradRatio(i,1)*walk.corrHelper.ParamGradient[j](i,1)+
                          2.*walk.corrHelper.GradRatio(i,2)*walk.corrHelper.ParamGradient[j](i,2));
      }
      double paramVal = corr._params[j];
      hamgrad[j] = abs(paramVal) < 1.e-10 ? 0 : hamgrad[j]/paramVal;
    }
  }
  
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
    
    
    const MatrixXd& thetaInv = walk.refHelper.thetaInv[0],
        &Laplacian = walk.refHelper.Laplacian[0],
        &AOLaplacian = walk.refHelper.AOLaplacian;
    MatrixXd AOGradx   = walk.refHelper.AOGradient[0],
        AOGrady   = walk.refHelper.AOGradient[1],
        AOGradz   = walk.refHelper.AOGradient[2];
    
    const array<MatrixXd, 3>& Gradient = walk.refHelper.Gradient;

    MatrixXd Gradx = Gradient[0];
    MatrixXd Grady = Gradient[1];
    MatrixXd Gradz = Gradient[2];
    for (int mo = 0; mo <nelec; mo++) {
      Gradx.row(mo) *= walk.corrHelper.GradRatio(mo,0);
      Grady.row(mo) *= walk.corrHelper.GradRatio(mo,1);
      Gradz.row(mo) *= walk.corrHelper.GradRatio(mo,2);
      AOGradx.row(mo) *= walk.corrHelper.GradRatio(mo,0);
      AOGrady.row(mo) *= walk.corrHelper.GradRatio(mo,1);
      AOGradz.row(mo) *= walk.corrHelper.GradRatio(mo,2);
    }

    
    //precalculate A-1 B A-1
    MatrixXd X = thetaInv * Laplacian * thetaInv;
    MatrixXd Xgx = thetaInv * Gradx * thetaInv;
    MatrixXd Xgy = thetaInv * Grady * thetaInv;
    MatrixXd Xgz = thetaInv * Gradz * thetaInv;

    
    for (int mo = 0; mo <nelec; mo++) {
      
      for (int orb = 0; orb < 2*norbs; orb++) {
        
        //laplacian contribution
        {
          double t1 = thetaInv.row(mo).dot(AOLaplacian.col(orb));
          double t2 = X       .row(mo).dot(AoRi       .col(orb));
          
          hamgrad[numVars + numDets + orb*nelec + mo] += -0.5*(t1 - t2);
        }


        //grad contribution
        {
          double t1 = thetaInv.row(mo).dot(AOGradx.col(orb));
          double t2 = Xgx     .row(mo).dot(AoRi   .col(orb));
          
          hamgrad[numVars + numDets + orb*nelec + mo] += -(t1 - t2);
        }

        {
          double t1 = thetaInv.row(mo).dot(AOGrady.col(orb));
          double t2 = Xgy     .row(mo).dot(AoRi   .col(orb));
          
          hamgrad[numVars + numDets + orb*nelec + mo] += -(t1 - t2);
        }

        {
          double t1 = thetaInv.row(mo).dot(AOGradz.col(orb));
          double t2 = Xgz     .row(mo).dot(AoRi   .col(orb));
          
          hamgrad[numVars + numDets + orb*nelec + mo] += -(t1 - t2);
        }

      }
      
    }
  }

  VectorXd grad = hamgrad;
  grad.setZero(); double factor = 1.0;
  OverlapWithGradient(const_cast<rWalker<rJastrow, rSlater>&>(walk), factor, grad);
  hamgrad += (-0.5*kineticTot+
              potentialij+
              potentiali) * grad;
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


  double kinetic = 0.0;
  
  {
    MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs


    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,0) * walk.refHelper.Gradient[0].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,1) * walk.refHelper.Gradient[1].row(i);
      Bij.row(i) += 2.*walk.corrHelper.GradRatio(i,2) * walk.refHelper.Gradient[2].row(i);
    }

    for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
      kinetic += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
      kinetic += walk.corrHelper.LaplaceRatio[i];
    }
  }
  return -0.5*(kinetic) + potentialij+potentiali;

}

