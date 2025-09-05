from pyscf import gto, scf
from pyscf.data import elements
import numpy as np
from math import ceil, floor


def run_atomic_scf(atom_symbol,basis,charge=0):

    if elements.charge(atom_symbol)%2 == 0 and charge%2 == 0:
        spin = 0
    elif elements.charge(atom_symbol)%2 != 0 and charge%2 != 0:
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


def run_atomic_scf_partial_charge(atom_symbol,basis,charge=0):

    if charge%1 == 0:
        D = run_atomic_scf(atom_symbol,basis,charge)
    elif charge%1 != 0:
        charge_up = ceil(charge)
        Dup = run_atomic_scf(atom_symbol,basis,charge_up)
        charge_down = floor(charge)
        Ddown = run_atomic_scf(atom_symbol,basis,charge_down)
        D = (charge_up-charge) * Dup + abs(charge_down-charge) * Ddown
    else:
        print('What are you doing?')

    return D


def generate_SAD_guess(atoms,coords,basis):

    mol = gto.M(atom=list(zip(atoms, coords)), basis=basis)
    nao = mol.nao

    ao_ids = mol.aoslice_by_atom()

    D0 = np.zeros((nao,nao))

    for iA, (_, _, p1, p2) in enumerate(ao_ids):
        
        dm_atom = run_atomic_scf(atoms[iA],basis)
        dm_atom = dm_atom[0] + dm_atom[1]
        D0[p1:p2, p1:p2] = dm_atom

    return D0

def generate_SAD_guess_partial_charge(atoms,coords,basis,charges):

    mol = gto.M(atom=list(zip(atoms, coords)), basis=basis)
    nao = mol.nao

    ao_ids = mol.aoslice_by_atom()

    D0 = np.zeros((nao,nao))

    for iA, (_, _, p1, p2) in enumerate(ao_ids):
        
        dm_atom = run_atomic_scf_partial_charge(atoms[iA],basis,charges[iA])
        dm_atom = dm_atom[0] + dm_atom[1]
        D0[p1:p2, p1:p2] = dm_atom

    return D0















