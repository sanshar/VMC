from pyscf.data import elements

def write_pp(mol):
    atms = mol.elements
    pp = mol._ecp
    #if empty dict don't print ppInfo.txt
    if pp is None or len(pp) == 0:
        return
    f = open('ppInfo.txt', 'w')
    for atm in pp:
        #atomic info
        pp_atm = pp[atm]
        z = elements.charge(atm)
        #number of core electrons
        ncore = pp_atm[0]
        f.write("%i\t%i\tidx" % (z, ncore))
        #indices of current atom in atom string
        n = [i for i, e in enumerate(atms) if e == atm]
        for i in n:
            f.write("\t%i" % i)
        f.write("\n")
        #angular momentum channels
        for l in range(len(pp_atm[1])):
            lchannel = pp_atm[1][l]
            f.write("%i\n" % lchannel[0])
            for p in range(len(lchannel[1])):
                power = lchannel[1][p]
                for g in power:
                    f.write("%i\t%.8f\t%.8f\n" % (p, g[0], g[1]))
    f.close()

