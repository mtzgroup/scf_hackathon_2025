import numpy as np
from terachem_util.data import atomic_numbers
from terachem_util.system import System
from rywang42.uhf_wrapper import UHF # this has the be updated
from math import ceil, floor


def run_atomic_scf(atom_symbol, basis, charge=0):
    if atomic_numbers[atom_symbol] % 2 == charge % 2:
        spin = 1
    else:
        spin = 2

    atom_mol = System.from_str(f"{atom_symbol} 0 0 0")
    mf_atom = UHF(atom_mol, basis=basis, charge=charge, multiplicity=spin)
    mf_atom.kernel()
    return mf_atom.make_rdm1()


def run_atomic_scf_partial_charge(atom_symbol, basis, charge=0):
    if charge % 1 == 0:
        D = run_atomic_scf(atom_symbol, basis, charge)
    else:
        charge_up = ceil(charge)
        Dup = run_atomic_scf(atom_symbol, basis, charge_up)
        Dup_sum = Dup[0] + Dup[1]
        charge_down = floor(charge)
        Ddown = run_atomic_scf(atom_symbol, basis, charge_down)
        Ddown_sum = Ddown[0] + Ddown[1]
        D = (
            float(charge_up - charge) * Dup_sum
            + float(abs(charge_down - charge)) * Ddown_sum
        )

    return D


def generate_SAD_guess(atoms, bi, basis):
    nao = bi.nao_cart
    ranges_cart = bi.ranges_cart

    D0 = np.zeros((nao, nao))

    for iA, ranges_atm in enumerate(ranges_cart):
        dm_atom = run_atomic_scf(atoms[iA], basis)
        dm_atom = dm_atom[0] + dm_atom[1]
        to_atom = np.concatenate(
            [np.arange(s.start, s.stop, dtype=np.intp) for s in ranges_atm]
        )
        D0[np.ix_(to_atom, to_atom)] = dm_atom

    return D0


def generate_SAD_guess_partial_charge(atoms, bi, basis, charges):
    nao = bi.nao_cart
    ranges_cart = bi.ranges_cart

    D0 = np.zeros((nao, nao))

    for iA, ranges_atm in enumerate(ranges_cart):
        dm_atom = run_atomic_scf_partial_charge(atoms[iA], basis, charges[iA])
        to_atom = np.concatenate(
            [np.arange(s.start, s.stop, dtype=np.intp) for s in ranges_atm]
        )
        D0[np.ix_(to_atom, to_atom)] = dm_atom

    return D0
