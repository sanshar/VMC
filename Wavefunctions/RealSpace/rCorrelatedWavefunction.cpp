#include "rCorrelatedWavefunction.h"

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::HamOverlap(const rWalker<rJastrow, rSlater>& walk,
                                                           Eigen::VectorXd& hamgrad) const {  
  const_cast<rWalker<rJastrow, rSlater>&>(walk).HamOverlap(ref, corr, hamgrad);  
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


  if (schd.hf == "ghf" ) {
    double kinetic = 0.0;

    {
      MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs
      for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
        Bij.row(i) += 2.*walk.corrHelper.GradRatio.row(i) * walk.refHelper.Gradient[i];
      }
      
      for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
        kinetic += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
        kinetic += walk.corrHelper.LaplaceRatio[i];
      }
    }
    return -0.5*(kinetic) + potentialij+potentiali;
  }
  else {
      
    double kinetica = 0.0;
    //Alpha
    {
      MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs
        
      for (int i=0; i<walk.d.nalpha; i++) 
        Bij.row(i) += 2.*walk.corrHelper.GradRatio.row(i) * walk.refHelper.Gradient[i];
        
      for (int i=0; i<walk.d.nalpha; i++) {
        kinetica += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
        kinetica += walk.corrHelper.LaplaceRatio[i];
      }
    }
      
    double kineticb = 0.0;
    //Beta
    if (walk.d.nbeta != 0)
    {
      MatrixXd Bij = walk.refHelper.Laplacian[1]; //i = nelec , j = norbs
      int nalpha = walk.d.nalpha;
        
      for (int i=0; i<walk.d.nbeta; i++) 
        Bij.row(i) += 2*walk.corrHelper.GradRatio.row(i+nalpha) * walk.refHelper.Gradient[i+nalpha];
        
      for (int i=0; i<walk.d.nbeta; i++) {
        kineticb += Bij.row(i).dot(walk.refHelper.thetaInv[1].col(i));
        kineticb += walk.corrHelper.LaplaceRatio[i+nalpha];
      }
    }
    return -0.5*(kinetica+kineticb) + potentialij+potentiali;
  }

}

