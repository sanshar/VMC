import numpy
from pyscf import lib
from pyscf import gto, scf, ao2mo, tools
import scipy.special


mol = gto.M(
atom = '''Mg 0 0 0;''',
unit = "Bohr",
cart = True,
basis = 'bfd-vdz',
ecp = 'bfd-pp',
spin = 0,
verbose = 5
)

gto.write_gto(mol)
gto.write_pp(mol)

mf = scf.RHF(mol)
#mf = scf.GHF(mol)
mf.kernel()

norbs = mf.mo_coeff.shape[0]
fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(mf.mo_coeff[i,j]))
    fileHF.write('\n')
