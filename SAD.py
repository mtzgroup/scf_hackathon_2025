from pyscf import gto, scf
from pyscf.data import elements
import numpy as np


def run_atomic_scf(atom_symbol,basis,charge=0):

    if elements.charge(atom_symbol)%2 == 0 and charge%2 == 0:
        spin = 0
    else:
        spin = 1

    atom_mol = gto.Mole(atom=[[atom_symbol, (0, 0, 0)]], 
                        basis = basis, 
                        charge = charge,
                        spin = spin)   
    atom_mol.build()
    mf_atom = scf.UHF(atom_mol)
    mf_atom.kernel()
    
    return mf_atom.make_rdm1()

def generate_SAD_guess(atoms,coords,basis,restricted=True):

    mol = gto.M(atom=list(zip(atoms, coords)), basis=basis)
    nao = mol.nao

    ao_ids = mol.aoslice_by_atom()

    D0 = np.zeros((nao,nao))

    for iA, (_, _, p1, p2) in enumerate(ao_ids):
        
        dm_atom = run_atomic_scf(atoms[iA],basis)
        dm_atom = dm_atom[0] + dm_atom[1]
        D0[p1:p2, p1:p2] = dm_atom

    return D0















