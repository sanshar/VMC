#!/usr/bin/env python

import os
import numpy as np
from libdmet.system import integral
from libdmet.solver import scf_solver
from libdmet.solver import cc
from libdmet.system import lattice
from libdmet_solid.utils.misc import read_poscar
import h5py

def write_dqmc(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol[0].shape[0]])
        fh5['hcore_up'] = hcore[0].flatten()
        fh5['hcore_dn'] = hcore[1].flatten()
        fh5['hcore_mod_up'] = hcore_mod[0].flatten()
        fh5['hcore_mod_dn'] = hcore_mod[1].flatten()
        fh5['chol_up'] = chol[0].flatten()
        fh5['chol_dn'] = chol[1].flatten()
        fh5['energy_core'] = enuc


def writeMat(mat, fileName, isComplex=False):
  fileh = open(fileName, 'w')
  for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        if (isComplex):
          fileh.write('(%16.10e, %16.10e) '%(mat[i,j].real, mat[i,j].imag))
        else:
          fileh.write('%16.10e '%(mat[i,j]))
      fileh.write('\n')
  fileh.close()


hf = h5py.File('ImpHam.h5', 'r')
ImpHam = integral.load("ImpHam.h5")
nocc = 30
norb = 60

dm0 = np.load("dm0.npy")
#labels = np.load("lo_labels.npy")

solver = scf_solver.SCFSolver(restricted=False, tol=1e-10, scf_newton=False)
rho, E = solver.run(ImpHam, nelec=nocc*2, dm0=dm0)
mo_coeff = solver.scfsolver.mf.mo_coeff

# expressing eri's as sum of squares
eri = ImpHam.H2['ccdd']
h1 = ImpHam.H1['cd']
enuc = float(ImpHam.H0)
block_eri = np.block([[eri[0], eri[2]], [eri[2].T, eri[1]]])
evals, evecs = np.linalg.eigh(block_eri)
nchol = (evals > 1.e-8).nonzero()[0].shape[0]
evals_sqrt = np.sqrt(evals[ evals > 1.e-8 ])
chol = np.zeros((2, nchol, norb, norb))
for i in range(nchol):
  for m in range(norb):
    for n in range(m+1):
      triind = m*(m+1)//2 + n
      chol[0, i, m, n] = evals_sqrt[-i-1] * evecs[triind, -i-1]
      chol[0, i, n, m] = evals_sqrt[-i-1] * evecs[triind, -i-1]
      chol[1, i, m, n] = evals_sqrt[-i-1] * evecs[norb*(norb+1)//2 + triind, -i-1]
      chol[1, i, n, m] = evals_sqrt[-i-1] * evecs[norb*(norb+1)//2 + triind, -i-1]

# checking uhf energy with cholesky ints
coul = np.einsum('sgpr,spr->g', chol, rho)
exc = np.einsum('sgpr,spt->sgrt', chol, rho)
e2 = (np.einsum('g,g->', coul, coul) - np.einsum('sgtr,sgrt->', exc, exc) )/2
e1 = np.einsum('ij,ji->', h1[0], rho[0]) + np.einsum('ij,ji->', h1[1], rho[1])
print(f'uhf ene from chol: {enuc + e1 + e2}')

# writing afqmc ints
nbasis = norb
v0_up = 0.5 * np.einsum('nik,njk->ij', chol[0], chol[0], optimize='optimal')
v0_dn = 0.5 * np.einsum('nik,njk->ij', chol[1], chol[1], optimize='optimal')
h1_mod = [ h1[0] - v0_up, h1[1] - v0_dn ]
chol_flat = [ chol[0].reshape((nchol, -1)), chol[1].reshape((nchol, -1)) ]
write_dqmc(h1, h1_mod, chol_flat, 2*nocc, nbasis, enuc, filename='FCIDUMP_chol')

uhfCoeffs = np.empty((norb, 2*norb))
uhfCoeffs[::,:norb] = mo_coeff[0]
uhfCoeffs[::,norb:] = mo_coeff[1]
writeMat(uhfCoeffs, "uhf.txt")

# cc

#solver = cc.CCSD(restricted=False, tol=1e-10, scf_newton=False)
#rho_cc, E_cc = solver.run(ImpHam, nelec=nocc*2, dm0=dm0, ccsdt=True, ccsdt_energy=True)

exit(0)

cell = read_poscar(fname="./CCO-AFM-frac.vasp")
cell.basis = 'gth-szv-mol-opt-sr'
cell.pseudo = 'gth-pade'
kmesh = [6, 6, 1]
cell.build()

Lat = lattice.Lattice(cell, kmesh)
Lat.mulliken_lo_R0(rho[:, :norb-nocc, :norb-nocc], labels=np.asarray(labels))
