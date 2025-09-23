import numpy as np
from terachem_util.data import atomic_numbers
from systems import System
from UHF import UHF # this has the be updated
from math import ceil, floor

from terachem import intbox
from basis import populate_basis_shells
from terachem_util import use

S_dd = np.array([
    [1,   0,   0,   0,     0,     0],
    [0,   1,   0,   0,     0,     0],
    [0,   0,   1,   0,     0,     0],
    [0,   0,   0,   1.0,   1/3.0, 1/3.0],
    [0,   0,   0,   1/3.0, 1.0,   1/3.0],
    [0,   0,   0,   1/3.0, 1/3.0, 1.0]
])

S_dd_inv = np.array([
    [1,   0,   0,    0,     0,     0],
    [0,   1,   0,    0,     0,     0],
    [0,   0,   1,    0,     0,     0],
    [0,   0,   0,    1.2,  -0.3,  -0.3],
    [0,   0,   0,   -0.3,   1.2,  -0.3],
    [0,   0,   0,   -0.3,  -0.3,   1.2]
])

U_dd = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, np.sqrt(3)/2, -0.5, 1/np.sqrt(5)],
        [0, 0, 0, -np.sqrt(3)/2, -0.5, 1/np.sqrt(5)],
        [0, 0, 0, 0, 1, 1/np.sqrt(5)]
    ])


def spherical_average(dm_atom, mol_atom):
    """
    Apply spherical averaging to an atomic density matrix.
    dm_atom : (nao,nao) total atomic density matrix
    mol_atom : gto.Mole for the atom
    """
    dm_sph = dm_atom.copy()

    p0 = 0
    l = 0
    for shell in mol_atom.bi.shells_by_angmom:
        nctr = len(shell)
        if nctr == 0:
            break

        nbas = int((l+1)*(l+2)/2)

        for i in range(nctr):        
            p1 = p0 + nbas

            if l == 0: #do nothing
                pass
            elif l == 1: # p orbitals
                block = dm_atom[p0:p1, p0:p1]
                N = np.trace(block).real
                block_sph = (N / nbas) * np.eye(nbas)
                dm_sph[p0:p1, p0:p1] = block_sph
            elif l == 2: # d orbitals - doing the same as terachem
                block = dm_atom[p0:p1, p0:p1]
                pop = block @ S_dd
                dm_sph_like = U_dd @ pop @ U_dd.T
                N = np.trace(dm_sph_like).real / 5                
                dm_sph_like_avg = np.eye(nbas)*N
                dm_sph_like_avg[-1,-1] = dm_sph_like[-1,-1]
                block_sph = S_dd @ U_dd @ dm_sph_like_avg @ U_dd.T
                dm_sph[p0:p1, p0:p1] = block_sph
            else:
                print('Wrong value of l: only s-, p-, and d-orbitals work')
            p0 = p1
        l+=1
    return dm_sph


def run_atomic_scf(atom_symbol, basis, charge=0):
    if atomic_numbers[atom_symbol] % 2 == charge % 2:
        spin = 1
    else:
        spin = 2

    atom_mol = System.from_str(f"{atom_symbol} 0 0 0").with_(charge=charge, spinmult=spin)
    mf_atom = UHF(atom_mol, basis=basis)
    mf_atom.kernel()
    D = mf_atom.make_rdm1()
    return spherical_average(D[0]+D[1],mf_atom)


def run_atomic_scf_partial_charge(atom_symbol, basis, charge=0):
    if charge % 1 == 0:
        D = run_atomic_scf(atom_symbol, basis, charge)
    else:
        charge_up = ceil(charge)
        Dup = run_atomic_scf(atom_symbol, basis, charge_up)
        charge_down = floor(charge)
        Ddown = run_atomic_scf(atom_symbol, basis, charge_down)
        D = (
            float(charge_up - charge) * Dup
            + float(abs(charge_down - charge)) * Ddown
        )

    return D


def generate_SAD_guess(atoms, bi, basis):
    nao = bi.nao_cart
    ranges_cart = bi.ranges_cart

    D0 = np.zeros((nao, nao))

    for iA, ranges_atm in enumerate(ranges_cart):
        dm_atom = run_atomic_scf(atoms[iA], basis)
        to_atom = np.concatenate(ranges_atm)
        D0[np.ix_(to_atom, to_atom)] = dm_atom

    return D0


def generate_SAD_guess_partial_charge(atoms, basis, charges):

    with use(intbox):
        # Setup basis and integrals
        bi = populate_basis_shells(intbox, basis, atoms)
        
        nao = bi.nao_cart
        ranges_cart = bi.ranges_atom_cart

    D0 = np.zeros((nao, nao))

    for iA, ranges_atm in enumerate(ranges_cart):
        dm_atom = run_atomic_scf_partial_charge(atoms[iA], basis, charges[iA])
        to_atom = np.concatenate(ranges_atm)
        D0[np.ix_(to_atom, to_atom)] = dm_atom

    return D0
