#!/usr/bin/env python

import numpy as np
from scipy import linalg as la
from pyscf import gto, scf, ao2mo, lo, fci
import h5py

np.set_printoptions(3, linewidth=1000, suppress=True)

mol = gto.M(
atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
''',
basis = '631g',
charge = 1,
spin = 1,  # = 2S = spin_up - spin_down
verbose = 4
)

rmf = scf.RHF(mol)
rmf.kernel()

cisolver = fci.FCI(rmf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)
dm1_fci = cisolver.make_rdm1(ci, mol.nao, mol.nelec)
h1e = rmf.mo_coeff.T.dot(rmf.get_hcore()).dot(rmf.mo_coeff)
e1_fci = np.trace(np.dot(dm1_fci, h1e))

mf = scf.UHF(mol)
mf.kernel()

H0 = mf.energy_nuc()
ovlp = mf.get_ovlp()
hcore = mf.get_hcore()
eri = mf._eri
dm = mf.make_rdm1()

def scdm(coeff, ovlp):
    aux = lo.orth.lowdin(ovlp)
    no = coeff.shape[1]
    ova = coeff.T @ ovlp @ aux
    piv = la.qr(ova, pivoting=True)[2]
    bc = ova[:, piv[:no]]
    ova = np.dot(bc.T, bc)
    s12inv = lo.orth.lowdin(ova)
    return coeff @ bc @ s12inv

C = np.asarray([scdm(mo, ovlp) for mo in mf.mo_coeff])

from libdmet.utils import mdot
from libdmet.system import integral
from libdmet.basis_transform import make_basis
from libdmet.solver import scf_solver

h1 = np.asarray([mdot(lo.conj().T, hcore, lo) for lo in C])
h2_aa = ao2mo.kernel(eri, C[0])
h2_bb = ao2mo.kernel(eri, C[1])
h2_ab = ao2mo.kernel(eri, (C[0], C[0], C[1], C[1]))
h2 = np.asarray([h2_aa, h2_bb, h2_ab])
#dm = make_basis.transform_rdm1_to_lo_mol(dm, C, ovlp)
dm = make_basis.transform_rdm1_to_lo(dm, C, ovlp)
norb = mol.nao_nr()
nelec = mf.nelec
H1 = {}
H1['cd'] = h1
H2 = {}
H2['ccdd'] = h2
ints = integral.Integral(norb, restricted=False, bogoliubov=False,
                         H0=H0, H1=H1, H2=H2, ovlp=None)

# save integrals
print ("norb: ", norb)
print ("nelec: ", nelec)
print ("E_uhf from pyscf", mf.e_tot)
ints.save("ImpHam.h5")
np.save("dm.npy", dm)

# re-compute uhf energy from UIHF
ImpHam = integral.load("ImpHam.h5")
nocc = nelec
norb = ImpHam.norb

dm0 = np.load("dm.npy")

solver = scf_solver.SCFSolver(restricted=False, tol=1e-10, scf_newton=False, Sz=nelec[0]-nelec[1])
rho, E = solver.run(ImpHam, nelec=sum(nelec), dm0=dm0)
Mo_coeff = solver.scfsolver.mf.mo_coeff

def write_dqmc(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    with h5py.File(filename, 'w') as fh5:
        print(np.array([nelec, nmo, ms, chol[0].shape[0]]))
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
e1_hf = e1

# writing afqmc ints
nbasis = norb
v0_up = 0.5 * np.einsum('nik,njk->ij', chol[0], chol[0], optimize='optimal')
v0_dn = 0.5 * np.einsum('nik,njk->ij', chol[1], chol[1], optimize='optimal')
h1_mod = [ h1[0] - v0_up, h1[1] - v0_dn ]
chol_flat = [ chol[0].reshape((nchol, -1)), chol[1].reshape((nchol, -1)) ]
write_dqmc(h1, h1_mod, chol_flat, sum(nelec), nbasis, enuc, ms=nelec[0] - nelec[1], filename='FCIDUMP_chol')

uhfCoeffs = np.empty((norb, 2*norb))
uhfCoeffs[::,:norb] = Mo_coeff[0]
uhfCoeffs[::,norb:] = Mo_coeff[1]
writeMat(uhfCoeffs, "uhf.txt")

# cc
from libdmet.solver import cc
solver = cc.CCSD(restricted=False, tol=1e-10, Sz=nelec[0]-nelec[1], scf_newton=False)
rho_cc, E_cc = solver.run(ImpHam, nelec=sum(nelec), dm0=dm0, ccsdt=True, ccsdt_energy=True)
e1_cc = np.einsum('ij,ji->', h1[0], rho_cc[0]) + np.einsum('ij,ji->', h1[1], rho_cc[1])

print('\n\n')
print(f'e1_hf: {e1_hf}')
print(f'e1_cc: {e1_cc}')
print(f'e1_fci: {e1_fci}')

# afqmc
import sys
import os

def write_afqmc_input(dt = 0.005, nsteps = 50, ndets = 100, fname = 'afqmc.json'):
  afqmc_input =  '''
{
  "system":
  {
    "integrals": "FCIDUMP_chol"
  },
  "wavefunction":
  {
    "right": "uhf",
    "left": "uhf",
    "ndets": %i
  },
  "sampling":
  {
    "seed": %i,
    "phaseless": true,
    "dt": %f,
    "nsteps": %i,
    "nwalk": 50,
    "choleskyThreshold": 2.0e-3,
    "orthoSteps": 20,
    "stochasticIter": 300
  },
  "print":
  {
    "scratchDir": "./scratch"
  }
}
  '''%(ndets, np.random.randint(1, 1e6), dt, nsteps)
  f = open (fname, "w")
  f.write(afqmc_input)
  f.close()
  return

# change these to your paths
dqmc_binary = '/projects/anma2640/VMC/dqmc_uihf/VMC/bin/DQMC'
blocking_script = '/projects/anma2640/VMC/dqmc/VMC/scripts/blocking.py'

nproc = 36
os.system("export OMP_NUM_THREADS=1; rm samples.dat -f")

write_afqmc_input()
command = f'''
              mpirun -np {nproc} {dqmc_binary} afqmc.json > afqmc.out;
              python {blocking_script} samples.dat 50 > blocking.out;
           '''
os.system(command)

# read afqmc rdm and calculate observables
import csv
import pandas as pd

# nobs is the number of observables
nobs = 1
constants = np.array([ 0. ])
fcount = [0, 0]
observables_afqmc =[ [ ], [ ] ]
weights = [ [ ], [ ] ]   # weights for up and down are the same
for (i, sz) in enumerate(['up', 'dn']):
 for filename in os.listdir():
   if (filename.startswith(f'rdm_{sz}')):
     fcount[i] += 1
     with open(filename) as fh:
       weights[i].append(float(fh.readline()))
     cols = list(range(norb))
     df = pd.read_csv(filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
     rdm_i = df.to_numpy()
     obs_i = constants.copy()
     for n in range(nobs):
       obs_i[n] += np.trace(np.dot(rdm_i, h1[i]))
     observables_afqmc[i].append(obs_i)

fcount = fcount[0]
weights = np.array(weights[0])
observables_afqmc = np.array(observables_afqmc[0]) + np.array(observables_afqmc[1])
obsMean = np.zeros(nobs)
obsError = np.zeros(nobs)
v1 = weights.sum()
v2 = (weights**2).sum()
for n in range(nobs):
  obsMean[n] = np.multiply(weights, observables_afqmc[:, n]).sum() / v1
  obsError[n] = (np.multiply(weights, (observables_afqmc[:, n] - obsMean[n])**2).sum() / (v1 - v2 / v1) / (fcount - 1))**0.5

obsVar = np.array([ e1_hf ])
np.set_printoptions(precision=7, linewidth=1000, suppress=True)
print(f'mixed obs_afqmc: {obsMean}')
print(f'extrapolated obs: {2*obsMean - obsVar}')
print(f'errors: {obsError}')
