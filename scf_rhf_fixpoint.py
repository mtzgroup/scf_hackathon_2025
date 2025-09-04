"""
Vanilla fixed-point RHF SCF implemented with TeraChem IntBox
"""

from terachem import intbox
from terachem_util import use, gpubox
from terachem_util.basis import populate_basis_shells
from terachem_util.classical import nn_repulsion
from terachem_util.data import atomic_numbers
from terachem_util.units import Bohr

import numpy as np
import scipy

from SAD import generate_SAD_guess

import logging

logging.basicConfig(level=logging.INFO)


# fmt: off
atm_names = ["H", "H", "O"]
coords = np.array([[-2.48126, 1.95909, 2.293070], [-2.60783, 2.63895, 0.902916], [-2.15230, 1.93581, 1.384490]])
# fmt: on
basis = "sto-3g"
pq_thre = 1e-15
threspdp = 0.0  # double precision
maxiter = 30
energy_conv = 1.0e-6
drms_conv = 1.0e-3

coords /= Bohr
charges = atomic_numbers[atm_names]
nocc = np.sum(charges) // 2
e_nuc = nn_repulsion(coords, charges)


def fock2density(fock, x):
    fockp = x @ fock @ x
    _, cp = np.linalg.eigh(fockp)
    c = x @ cp
    c_occ = c[:, :nocc]
    return (c_occ @ c_occ.T) * 2


def evaluate(hcore, fock, den, s, x):
    energy = np.einsum("pq,pq->", fock + hcore, den) / 2 + e_nuc
    diis_e = x @ (fock @ den @ s - s @ den @ fock) @ x
    drms = np.mean(diis_e**2) ** 0.5
    return energy, drms


with use(gpubox([0]), intbox):
    bi = populate_basis_shells(basis, atm_names)
    intbox.update_coors(pq_thre, coords)

    hcore = np.zeros((bi.nao_cart, bi.nao_cart))
    intbox.one_e_k_core(hcore)
    intbox.one_e_v_core(
        1.0, 0.0, 0.0, pq_thre, len(charges), np.c_[coords, charges], hcore
    )

    s = np.empty((bi.nao_cart, bi.nao_cart))
    intbox.overlap(s)
    x = np.linalg.inv(scipy.linalg.sqrtm(s))
    # den = fock2density(hcore, x)  # H core guess
    den_spher = generate_SAD_guess(atm_names,coords,basis)
    den = bi.spherical_to_cartesian_order2(den_spher)

    energy_prev = 0.0
    for step in range(maxiter):
        fock = np.copy(hcore)
        intbox.j_fock_gen(0.0, threspdp, 1.0, 0.0, 0.0, den, fock)
        intbox.k_fock_sym(0.0, threspdp, -0.5, 0.0, 0.0, den, fock)

        energy, drms = evaluate(hcore, fock, den, s, x)
        print(
            f"SCF iteration {step + 1:2d}:  E={energy:12.6f}  dE={energy - energy_prev:10.3e}  dRMS={drms:10.3e}"
        )
        if abs(energy - energy_prev) < energy_conv and drms < drms_conv:
            print("SCF converged")
            break

        energy_prev = energy
        den = fock2density(fock, x)

    else:
        print(f"SCF did not converge after {maxiter} iterations")
