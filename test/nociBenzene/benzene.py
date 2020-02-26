#!/usr/bin/env python

from pyscf import gto, scf, tools, ao2mo, tools, cc
from pyscf.vmcscf import vmcActiveActive as vmc
from pyscf.shciscf import shci
import numpy as np
import scipy.linalg
import time, os, sys
from pyscf.lo import pipek, boys

t0 = time.time()

atomstring = '''
C  0.000517 0.000000  0.000299
C  0.000517 0.000000  1.394692
C  1.208097 0.000000  2.091889
C  2.415677 0.000000  1.394692
C  2.415677 0.000000  0.000299
C  1.208097 0.000000 -0.696898
H -0.939430 0.000000 -0.542380
H -0.939430 0.000000  1.937371
H  1.208097 0.000000  3.177246
H  3.355625 0.000000  1.937371
H  3.355625 0.000000 -0.542380
H  1.208097 0.000000 -1.782255
'''
mol = gto.M(
    atom = atomstring,
    unit = 'angstrom',
    basis = 'ccpvdz',
    verbose = 4,
    symmetry= 0,
    spin = 0)

# Create HF molecule
mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.scf()

mycc = cc.CCSD(mf)
mycc.set(frozen = list(range(0,6))+list(range(36, mf.mo_coeff.shape[0]))).run()

#####HCI###
'''
ncore, nact = 2, 8
mch = shci.SHCISCF(mf, nact, 10)
mch.fcisolver.nPTiter = 0  # Turn off perturbative calc.
mch.fcisolver.sweep_iter = [0, 3]
mch.fcisolver.DoRDM = True
# Setting large epsilon1 thresholds highlights improvement from perturbation.
mch.fcisolver.sweep_epsilon = [1e-2, 1e-2]
mch.max_cycle_macro = 1
e_noPT = mch.mc1step()[0]
exit(0)
'''

#####VMC###
ncore, nact = 6, 30
occ =[[0,4,0]*3,[0,0,4]*3,[0,1,0]*3,[0,0,1]*3]

mch = vmc.VMCSCF(mf, ncore, nact, loc="ibo", stochasticIter = 800, maxIter=50)
mch.frozen =list(range(0,6))+list(range(36, mf.mo_coeff.shape[0]))
mch.internal_rotation = True
e_noPT = mch.mc2step()[0]
 
