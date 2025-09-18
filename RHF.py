"""
PySCF-style RHF wrapper for TeraChem IntBox calculations
"""

from terachem import intbox
from terachem_util import use, gpubox
from basis import populate_basis_shells
from terachem_util.classical import nn_repulsion
from systems import System
from terachem_util.linalg import invsqrtm_spd
from terachem_util.typing import NDArrayf64
from DIIS import (
    DIIS,
    orthogonal_diagonalize,
    build_density,
)

import numpy as np


class RHF:
    """
    Restricted Hartree-Fock SCF calculation wrapper

    Similar interface to PySCF's RHF class
    """

    def __init__(self, mol: System, basis: str):
        self.mol = mol
        self.basis = basis

        # SCF parameters (can be modified before calling kernel())
        self.conv_tol = 1.0e-8  # Energy convergence threshold
        self.conv_tol_grad = 1.0e-4  # DIIS error convergence threshold
        self.max_cycle = 500  # Maximum SCF iterations
        self.diis_space = 8  # DIIS subspace size
        self.verbose = 4  # Verbosity level

        # TeraChem parameters
        self.pq_thre = 1e-15
        self.threspdp = 0.0

        # Results
        self.converged = False
        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None

        # Internal variables
        self._setup_electrons()
        self._e_nuc = nn_repulsion(self.mol.geometry, self.mol.charges)

    def _setup_electrons(self):
        """Setup electron configuration"""
        mult = self.mol.multiplicity
        nelec_total = self.mol.n_alpha_electrons + self.mol.n_beta_electrons
        if mult != 1:
            raise ValueError("Inconsistent charge/multiplicity with nuclear charges.")
        self.nocc = nelec_total // 2
        self.nelec = nelec_total

    def _rhf_energy(self, hcore, fock, den):
        """Compute RHF energy"""
        return np.einsum("pq,pq->", fock + hcore, den) / 2 + self._e_nuc
    
    def _get_j_and_k(self, D):
        """Build J and K matrices"""
        nao = self.nao
        hcore_shape = (nao, nao)

        # Build Coulomb matrix J(Dtot)
        J_buf = np.zeros(hcore_shape)
        intbox.j_fock_gen(0.0, self.threspdp, 1.0, 0.0, 0.0, D, J_buf)

        # Build exchange matrices K(Da) and K(Db)
        K_buf = np.zeros(hcore_shape)
        intbox.k_fock_sym(0.0, self.threspdp, -0.5, 0.0, 0.0, D, K_buf)
        return J_buf, K_buf

    def _scf_cycle(
        self,
        cycle: int,
        D: NDArrayf64,
        hcore: NDArrayf64,
        s: NDArrayf64,
        x: NDArrayf64,
        diis: DIIS[list[NDArrayf64]],
    ):
        """Perform one SCF cycle"""
        J_buf, K_buf = self._get_j_and_k(D)

        # Build Fock matrices
        F = hcore + J_buf + K_buf  # K_buf already has -0.5 coefficient from intbox call
        
        # DIIS - compute error vector first
        diis_error = diis.compute_error(F, D, s, x)
        diis.add_iteration([F], diis_error, s, x)
        drms = diis.get_latest_drms()
        if cycle > 1:
            F = diis.extrapolate_fock()

        # Diagonalize Fock matrices
        eps, C = orthogonal_diagonalize(F, x)

        # Store molecular orbitals and occupations
        self.mo_energy = (eps)
        self.mo_coeff = (C)

        # Build new density matrices
        D_new = build_density(C, self.nocc, restricted=True)

        energy = self._rhf_energy(hcore, F, D_new)

        return energy, D_new, drms

    def kernel(self, dm0=None):
        """
        Run RHF SCF calculation

        Parameters:
        -----------
        dm0 : np.ndarray, optional
            Initial density matrix. If None, use core Hamiltonian guess.

        Returns:
        --------
        float
            Total RHF energy
        """
        if self.verbose >= 4:
            print(f"Starting RHF calculation for {self.mol.geometry.shape[0]} atoms")
            print(f"Basis: {self.basis}")
            print(
                f"Charge: {self.mol.net_charge}, Multiplicity: {self.mol.multiplicity}"
            )
            print(f"Electrons: {self.nelec} (occupied orbitals: {self.nocc})")
            print("-" * 60)

        with use(intbox):
            # Setup basis and integrals
            bi = populate_basis_shells(intbox, self.basis, self.mol.atm_names)
            intbox.update_coors(self.pq_thre, self.mol.geometry)

            nao = bi.nao_cart
            self.nao = nao

            # Core Hamiltonian
            hcore = np.zeros((nao, nao))
            intbox.one_e_k_core(hcore)
            xyzq = np.c_[self.mol.geometry, self.mol.charges]
            intbox.one_e_v_core(1.0, 0.0, 0.0, self.pq_thre, len(xyzq), xyzq, hcore)

            # Overlap matrix and orthogonalization
            s = np.empty((nao, nao))
            intbox.overlap(s)
            x = invsqrtm_spd(s)

            # Initial guess
            if dm0 is None:
                _, C0 = orthogonal_diagonalize(hcore, x)
                D = build_density(C0, self.nocc, restricted=True)
            else:
                D = dm0

            # Initialize DIIS
            diis = DIIS(max_vec=self.diis_space)

            energy, energy_prev = 0.0, 0.0
            self.converged = False

            for cycle in range(1, self.max_cycle + 1):
                energy, D, drms = self._scf_cycle(cycle, D, hcore, s, x, diis)
                if self.verbose >= 4:
                    print(
                        f"cycle= {cycle:2d} E= {energy:16.12f}  delta_E= {energy - energy_prev:9.2e}  |g|= {drms:9.2e}"
                    )

                if (
                    abs(energy - energy_prev) < self.conv_tol
                    and drms < self.conv_tol_grad
                ):
                    self.converged = True
                    if self.verbose >= 4:
                        print("converged SCF energy = %.12f" % energy)
                    break
                energy_prev = energy
            else:
                print(f"SCF did not converge after {self.max_cycle} cycles")

            # Occupation numbers
            occ = np.zeros(nao)
            occ[: self.nocc] = 2.0  # Doubly occupied
            self.mo_occ = occ
            # Store density matrix
            self._dm = D

            self.e_tot = energy

        return energy

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        """
        Make 1-electron density matrix

        Returns:
        --------
        np.ndarray
            Density matrix
        """
        if not hasattr(self, "_dm"):
            raise RuntimeError("SCF not yet run. Call kernel() first.")
        return self._dm
    def analyze(self):
        """Print analysis of the calculation"""
        if not self.converged:
            print("Warning: SCF did not converge")

        print("\n" + "=" * 60)
        print("RHF ANALYSIS")
        print("=" * 60)
        print(f"Total Energy:      {self.e_tot:.12f} Ha")
        print(f"Nuclear Repulsion: {self._e_nuc:.12f} Ha")
        print(f"Electronic Energy: {self.e_tot - self._e_nuc:.12f} Ha")
        print(f"Electrons:         {self.nelec}")
        print(f"Occupied orbitals: {self.nocc}")

        if self.mo_energy is not None:
            print(f"\nHOMO: {self.mo_energy[self.nocc - 1]:.6f} Ha")
            print(
                f"LUMO: {self.mo_energy[self.nocc]:.6f} Ha"
                if self.nocc < len(self.mo_energy)
                else "LUMO: N/A"
            )
            if self.nocc > 0:
                gap = (
                    self.mo_energy[self.nocc] - self.mo_energy[self.nocc - 1]
                    if self.nocc < len(self.mo_energy)
                    else float("inf")
                )
                print(f"HOMO-LUMO gap: {gap:.6f} Ha ({gap * 27.2114:.3f} eV)")


def example_usage():
    """Example usage of the RHF wrapper"""

    # Water molecule
    mol = System.from_str("""
    H -2.48126  1.95909  2.293070
    H -2.60783  2.63895  0.902916
    O -2.15230  1.93581  1.384490
    """).with_(charge=0, spinmult=1)

    # Create RHF object
    mf = RHF(mol, basis="6-31gs")

    # Set convergence criteria (optional)
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-4
    mf.max_cycle = 50

    # Run calculation
    with use(gpubox([0])):
        energy = mf.kernel()


    # Analyze results
    mf.analyze()

    # Access results
    print(f"\nConverged: {mf.converged}")
    print(f"Final energy: {energy:.12f} Ha")


if __name__ == "__main__":
    example_usage()
