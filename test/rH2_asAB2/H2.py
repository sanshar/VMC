import numpy
from pyscf import lib
from pyscf import gto, scf, ao2mo, tools
import scipy.special



mol = gto.M(
atom = '''H 0 0 0; H 0 0 1.4''',
unit = "Bohr",
cart = True,
basis = 'ccpvdz',
spin = 0,
verbose = 5
)

gto.write_gto(mol)
gto.write_pp(mol)

mf = scf.RHF(mol)
mf.kernel()

asAO = mol.search_ao_label("H 1s")
f = open("asAO.txt", 'w')
for i in range(len(asAO)):
    f.write(f'{asAO[i]}\t')

norbs = mf.mo_coeff.shape[0]
fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(mf.mo_coeff[i,j]))
    fileHF.write('\n')
