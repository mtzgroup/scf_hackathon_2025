"""
PySCF-style UHF wrapper for TeraChem IntBox calculations
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


class UHF:
    """Unrestricted Hartree-Fock SCF calculation wrapper.
    Similar interface to PySCF's UHF class.
    """

    def __init__(self, mol: System, basis: str):
        self.mol = mol
        self.basis = basis

        # SCF parameters (can be modified before calling kernel())
        self.conv_tol = 1.0e-8  # Energy convergence threshold
        self.conv_tol_grad = 1.0e-4  # DIIS error convergence threshold
        self.max_cycle = 60  # Maximum SCF iterations
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
        self.nalpha = self.mol.n_alpha_electrons
        self.nbeta = self.mol.n_beta_electrons
        if self.nalpha < 0 or self.nbeta < 0 or (self.nalpha - self.nbeta) != mult - 1:
            raise ValueError("Inconsistent charge/multiplicity with nuclear charges.")

    def _uhf_energy(self, hcore, J, K_alpha, K_beta, Da, Db):
        """Compute UHF energy"""
        Dtot = Da + Db
        e1 = np.einsum("pq,pq->", hcore, Dtot)
        eJ = 0.5 * np.einsum("pq,pq->", J, Dtot)
        eK = -0.5 * (
            np.einsum("pq,pq->", K_alpha, Da) + np.einsum("pq,pq->", K_beta, Db)
        )
        return e1 + eJ + eK + self._e_nuc

    def _get_j_and_k(self, Da, Db):
        """Build J and K matrices"""
        Dtot = Da + Db
        nao = self.nao
        hcore_shape = (nao, nao)

        # Build Coulomb matrix J(Dtot)
        J_buf = np.zeros(hcore_shape)
        intbox.j_fock_gen(0.0, self.threspdp, 1.0, 0.0, 0.0, Dtot, J_buf)

        # Build exchange matrices K(Da) and K(Db)
        K_alpha = np.zeros(hcore_shape)
        intbox.k_fock_sym(0.0, self.threspdp, 1.0, 0.0, 0.0, Da, K_alpha)
        K_beta = np.zeros(hcore_shape)
        intbox.k_fock_sym(0.0, self.threspdp, 1.0, 0.0, 0.0, Db, K_beta)

        return J_buf, K_alpha, K_beta

    def _scf_cycle(
        self,
        cycle: int,
        Da: NDArrayf64,
        Db: NDArrayf64,
        hcore: NDArrayf64,
        s: NDArrayf64,
        x: NDArrayf64,
        diis: DIIS[list[NDArrayf64]],
    ):
        """Perform one SCF cycle"""
        J_buf, K_alpha, K_beta = self._get_j_and_k(Da, Db)

        # Build Fock matrices
        F_alpha = hcore + J_buf - K_alpha
        F_beta = hcore + J_buf - K_beta

        energy = self._uhf_energy(hcore, J_buf, K_alpha, K_beta, Da, Db)

        # DIIS
        diis.add_iteration([F_alpha, F_beta], [Da, Db], s, x)
        drms = diis.get_latest_drms()
        if cycle > 1:
            F_alpha, F_beta = diis.extrapolate_fock()

        # Diagonalize Fock matrices
        eps_alpha, Ca = orthogonal_diagonalize(F_alpha, x)
        eps_beta, Cb = orthogonal_diagonalize(F_beta, x)
        # Store molecular orbitals and occupations
        self.mo_energy = (eps_alpha, eps_beta)
        self.mo_coeff = (Ca, Cb)

        # Build new density matrices
        Da_new = build_density(Ca, self.nalpha)
        Db_new = build_density(Cb, self.nbeta)

        return energy, Da_new, Db_new, drms

    def kernel(self, dm0=None):
        """Run UHF SCF calculation

        Parameters:
        -----------
        dm0 : tuple of np.ndarray, optional
            Initial density matrices (Da, Db). If None, use core Hamiltonian guess.

        Returns:
        --------
        float
            Total UHF energy
        """
        if self.verbose >= 4:
            print(f"Starting UHF calculation for {self.mol.geometry.shape[0]} atoms")
            print(f"Basis: {self.basis}")
            print(
                f"Charge: {self.mol.net_charge}, Multiplicity: {self.mol.multiplicity}"
            )
            print(f"Electrons: α={self.nalpha}, β={self.nbeta}")
            print("-" * 60)

        with use(intbox):
            # Setup basis and integrals
            bi = populate_basis_shells(intbox, self.basis, self.mol.atm_names)
            self.bi = bi
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
                Ca0 = C0[:, : self.nalpha]
                Cb0 = C0[:, : self.nbeta]
                Da = Ca0 @ Ca0.T
                Db = Cb0 @ Cb0.T
            else:
                Da = 0.5*dm0
                Db = 0.5*dm0

            diis = DIIS(max_vec=self.diis_space)
            energy, energy_prev = 0.0, 0.0
            self.converged = False
            for cycle in range(1, self.max_cycle + 1):
                energy, Da, Db, drms = self._scf_cycle(cycle, Da, Db, hcore, s, x, diis)
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
        occ_alpha = np.zeros(nao)
        occ_beta = np.zeros(nao)
        occ_alpha[: self.nalpha] = 1.0
        occ_beta[: self.nbeta] = 1.0
        self.mo_occ = (occ_alpha, occ_beta)

        # Store density matrices
        self._dm = (Da, Db)

        self.e_tot = energy
        return energy

    def make_rdm1(self):
        """Make 1-electron density matrix

        Returns:
        --------
        tuple of np.ndarray
            (Da, Db) density matrices
        """
        if not hasattr(self, "_dm"):
            raise RuntimeError("SCF not yet run. Call kernel() first.")
        return self._dm

    def analyze(self):
        """Print analysis of the calculation"""
        if not self.converged:
            print("Warning: SCF did not converge")

        print("\n" + "=" * 60)
        print("UHF ANALYSIS")
        print("=" * 60)
        print(f"Total Energy:      {self.e_tot:.12f} Ha")
        print(f"Nuclear Repulsion: {self._e_nuc:.12f} Ha")
        print(f"Electronic Energy: {self.e_tot - self._e_nuc:.12f} Ha")
        print(f"Alpha electrons:   {self.nalpha}")
        print(f"Beta electrons:    {self.nbeta}")
        print(f"Total electrons:   {self.nalpha + self.nbeta}")

        if self.mo_energy is not None:
            eps_alpha, eps_beta = self.mo_energy
            print(f"\nHOMO(α): {eps_alpha[self.nalpha - 1]:.6f} Ha")
            print(
                f"LUMO(α): {eps_alpha[self.nalpha]:.6f} Ha"
                if self.nalpha < len(eps_alpha)
                else "LUMO(α): N/A"
            )
            print(
                f"HOMO(β): {eps_beta[self.nbeta - 1]:.6f} Ha"
                if self.nbeta > 0
                else "HOMO(β): N/A"
            )
            print(
                f"LUMO(β): {eps_beta[self.nbeta]:.6f} Ha"
                if self.nbeta < len(eps_beta)
                else "LUMO(β): N/A"
            )


def example_usage():
    """Example usage of the UHF wrapper"""

    # OH radical (doublet)
    mol = System.from_str("""
    O    0.000000   0.000000   0.000000
    H    0.000000   0.000000   0.969700
    """).with_(charge=0, spinmult=2)

    # Create UHF object
    mf = UHF(mol, basis="6-31gs")

    # Set convergence criteria (optional)
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-4
    mf.max_cycle = 60

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
