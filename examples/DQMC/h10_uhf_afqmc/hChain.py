import numpy as np
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
from pyscf.shciscf import shci
import sys, os
import scipy.linalg as la
import QMCUtils
import fcidump_rel
import h5py

# these need to be provided
nproc = 10
dice_binary = "/projects/anma2640/relDice/Dice/ZDice2"
vmc_root = "/projects/anma2640/VMC/dqmc_uihf/VMC/"

r = 1.8
atomstring = ""
for i in range(10):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose = 4,
    unit = 'bohr',
    symmetry = 0,
    spin = 0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# uhf
dm = [np.zeros((norb, norb)), np.zeros((norb, norb))]
for i in range(norb//2):
  dm[0][2*i, 2*i] = 1.
  dm[1][2*i+1, 2*i+1] = 1.
umf = scf.UHF(mol)
umf.kernel(dm)

# fci
cisolver = fci.FCI(mf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)
dm1_fci = cisolver.make_rdm1(ci, mol.nao, mol.nelec)
h1e = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
e1_fci = np.trace(np.dot(dm1_fci, h1e))

# dice
# writing input and integrals
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, mol.nao, mol.nelectron)
mc.mo_coeff = mf.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.davidsonTol = 5.e-5
mc.fcisolver.dE = 1.e-6
mc.fcisolver.maxiter = 6
mc.fcisolver.nPTiter = 0
mc.fcisolver.DoRDM = False
#shci.dryrun(mc, mc.mo_coeff)
shci.writeSHCIConfFile(mc.fcisolver, mol.nelec, False)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
  fh.write("readText\n")
  fh.write("writebestdeterminants 1000\n")


ughf = la.block_diag(umf.mo_coeff[0], umf.mo_coeff[1])
ughf = ughf[:, [ i//2 if (i%2 == 0) else norb + i//2 for i in range(2*norb)]]
ghcore = la.block_diag(umf.get_hcore(), umf.get_hcore())
h1g = ughf.T.dot(ghcore).dot(ughf) + 0.j
eri = ao2mo.kernel(mol, ughf[[ i//2 if (i%2 == 0) else norb + i//2 for i in range(2*norb)],:] + 0.j, intor='int2e_spinor')
fcidump_rel.from_integrals('FCIDUMP', h1g, eri, 2*norb, mol.nelectron, nuc=mf.energy_nuc())

# run dice calculation
print("Starting Dice calculation")
command = f"mpirun -np {nproc} {dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)
print("Finished Dice calculation\n")

# afqmc

print("Preparing AFQMC calculation")
# expressing eri's as sum of squares
h1_ao = umf.get_hcore()
h1 = [ umf.mo_coeff[0].T.dot(h1_ao).dot(umf.mo_coeff[0]), umf.mo_coeff[1].T.dot(h1_ao).dot(umf.mo_coeff[1])  ]
eriUp = ao2mo.kernel(umf._eri, umf.mo_coeff[0])
eriDn = ao2mo.kernel(umf._eri, umf.mo_coeff[1])
eriUpDn = ao2mo.incore.general(umf._eri, (umf.mo_coeff[0], umf.mo_coeff[0], umf.mo_coeff[1],umf.mo_coeff[1]))
enuc = mf.energy_nuc()
block_eri = np.block([[eriUp, eriUpDn], [eriUpDn.T, eriDn]])
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
rho = [ np.zeros((norb, norb)), np.zeros((norb, norb)) ]
for i in range(mol.nelec[0]):
  rho[0][i, i] = 1.
for i in range(mol.nelec[1]):
  rho[1][i, i] = 1.
coul = np.einsum('sgpr,spr->g', chol, rho)
exc = np.einsum('sgpr,spt->sgrt', chol, rho)
e2 = (np.einsum('g,g->', coul, coul) - np.einsum('sgtr,sgrt->', exc, exc) )/2
e1 = np.einsum('ij,ji->', h1[0], rho[0]) + np.einsum('ij,ji->', h1[1], rho[1])
print(f'uhf ene from chol: {enuc + e1 + e2}\n')
e1_hf = e1

# one-body energy
print(f'e1_hf: {e1_hf}')
print(f'e1_fci: {e1_fci}')

# writing afqmc ints
nbasis = norb
v0_up = 0.5 * np.einsum('nik,njk->ij', chol[0], chol[0], optimize='optimal')
v0_dn = 0.5 * np.einsum('nik,njk->ij', chol[1], chol[1], optimize='optimal')
h1_mod = [ h1[0] - v0_up, h1[1] - v0_dn ]
chol_flat = [ chol[0].reshape((nchol, -1)), chol[1].reshape((nchol, -1)) ]
QMCUtils.write_dqmc_uihf(h1, h1_mod, chol_flat, mol.nelectron, nbasis, enuc, filename='FCIDUMP_chol')

# write hf wave function coefficients
uhfCoeffs = np.empty((norb, 2*norb))
uhfCoeffs[::,:norb] = np.eye(norb)
uhfCoeffs[::,norb:] = np.eye(norb)
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_* -f")

# uhf trial
QMCUtils.write_afqmc_input(seed=4321, left="uhf", right="uhf", nwalk=20, stochasticIter=500, choleskyThreshold=1.e-3, fname="afqmc_uhf.json")
print("\nStarting AFQMC / UHF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_uhf.json > afqmc_uhf.out;
              mv samples.dat samples_uhf.dat
              python {blocking_script} samples_uhf.dat 100 > blocking_uhf.out;
              cat blocking_uhf.out;
           '''
os.system(command)
obsVar = np.array([ e1_hf ])
obsMean, obsError = QMCUtils.calculate_observables_uihf([ h1 ])
np.set_printoptions(precision=7, linewidth=1000, suppress=True)
print(f'\nmixed obs_afqmc: {obsMean}')
print(f'extrapolated obs: {2*obsMean - obsVar}')
print(f'errors: {obsError}')
print("Finished AFQMC / UHF calculation\n")

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_* -f")

# hci trial
for ndets in [ 10, 100 ]:
  QMCUtils.write_afqmc_input(left="multislater", right="uhf", nwalk=20, stochasticIter=500, ndets=ndets, choleskyThreshold=1.e-3, fname=f"afqmc_{ndets}.json")
  print(f"Starting AFQMC / HCI ({ndets}) calculation", flush=True)
  command = f'''
                mpirun -np {nproc} {afqmc_binary} afqmc_{ndets}.json > afqmc_{ndets}.out;
                mv samples.dat samples_{ndets}.dat
                python {blocking_script} samples_{ndets}.dat 100 > blocking_{ndets}.out;
                cat blocking_{ndets}.out;
             '''
  os.system(command)
  obsMean, obsError = QMCUtils.calculate_observables_uihf([ h1 ])
  np.set_printoptions(precision=7, linewidth=1000, suppress=True)
  print(f'\nmixed obs_afqmc: {obsMean}')
  print(f'errors: {obsError}')
  print(f"Finished AFQMC / HCI ({ndets}) calculation\n")

