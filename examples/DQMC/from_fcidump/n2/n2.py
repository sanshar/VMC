# N2 ground state

import os
import QMCUtils
import numpy as np

# these need to be provided
dice_binary = "/projects/joku8258/software/Dice/Dice"
vmc_root = "/projects/joku8258/QMC/VMC/"

nelec = 14
norb_frozen = 2
norb_act = 8
nelec_act = 10
norb_core = (nelec-nelec_act)//2 -norb_frozen # mc.ncore
spin = 0 #alpha-beta
nproc = 10
################################   
############ MeanField###############
QMCUtils.prepAFQMC_fromFCIDUMP(seed = 89649,left='rhf',spin=spin,norb_frozen=norb_frozen,nblocks=1000,fcidump="FCIDUMP",fname='afqmc.json')
#
############################################################################
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

print("Starting AFQMC / MF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc.json > afqmc.out;
              mv samples.dat samples_afqmc.dat
              python {blocking_script} samples_afqmc.dat 50 > blocking_afqmc.out;
              cat blocking_afqmc.out;
           '''
os.system(command)
print("Finished AFQMC / MF calculation")
#############################################################################

#### Multislater #############
QMCUtils.prepAFQMC_fromFCIDUMP(seed = 89649,left = 'multislater',spin=spin,norb_core=norb_core,norb_act=norb_act,nelec_act=nelec_act,norb_frozen=norb_frozen,nblocks=1000,ndets=100,fcidump="FCIDUMP",fname='afqmc_multislater.json')

#Assumes that the determinants are present in the folder as dets.bin (nroot=0) or dets_{nroot}.bin (nroot>0)
############################################################################
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

print("Starting AFQMC / HCI calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_multislater.json > afqmc_multislater.out;
              mv samples.dat samples_multislater.dat
              python {blocking_script} samples_multislater.dat 50 > blocking_multislater.out;
              cat blocking_multislater.out;
           '''
os.system(command)
print("Finished AFQMC / HCI calculation")
################################################################################









