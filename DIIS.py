"""
Direct Inversion of the Iterative Subspace (DIIS) accelerator for SCF calculations
"""

import numpy as np
from scipy.linalg import solve
from attrs import define, field

from terachem_util.typing import NDArrayf64

from collections import deque
import logging


def orthogonal_diagonalize(fock, x):
    """Return (eps, C) from orthogonalization matrix x (S^-1/2)."""
    fockp = x @ fock @ x
    eps, cp = np.linalg.eigh(fockp)
    c = x @ cp
    return eps, c


def build_density(C, nocc, restricted=False):
    """Build density from occupied orbitals."""
    C_occ = C[:, :nocc]
    return C_occ @ C_occ.T if not restricted else (C_occ @ C_occ.T) * 2


@define
class DIIS[FT: NDArrayf64 | list[NDArrayf64]]:
    """Direct Inversion of the Iterative Subspace (DIIS) accelerator

    Handles both RHF (single Fock matrix) and UHF (alpha/beta Fock matrices) cases
    by accepting lists of Fock matrices and error vectors.
    """

    max_vec: int = field(default=8)
    fock_history: deque[FT] = field(init=False)
    error_history: deque[FT] = field(init=False)
    n_fock: int | None = None  # Number of Fock matrices (1 for RHF, 2 for UHF)

    def __attrs_post_init__(self):
        self.fock_history = deque(maxlen=self.max_vec)
        self.error_history = deque(maxlen=self.max_vec)

    @staticmethod
    def compute_error(fock, dm, s, x):
        return x @ (fock @ dm @ s - s @ dm @ fock) @ x

    def add_iteration(
        self,
        focks: FT,
        dms: FT,
        s: NDArrayf64,
        x: NDArrayf64,
    ):
        focks = self._wrap_list(focks)
        dms = self._wrap_list(dms)
        if len(focks) != len(dms):
            raise ValueError(
                "Number of Fock matrices must match number of error vectors"
            )
        if self.n_fock is None:
            self.n_fock = len(focks)
        elif self.n_fock != len(focks):
            raise ValueError(
                f"Inconsistent number of Fock matrices: expected {self.n_fock}, got {len(focks)}"
            )

        self.fock_history.append([f.copy() for f in focks])
        errors = [self.compute_error(fock, dm, s, x) for fock, dm in zip(focks, dms)]
        self.error_history.append(errors)

    def get_latest_drms(self):
        return max(np.mean(e**2) ** 0.5 for e in self.error_history[-1])

    def extrapolate_fock(self) -> FT:
        n = len(self.error_history)
        if n < 2:
            # Not enough history for DIIS, return most recent Fock(s)
            return self._unwrap_list(self.fock_history[-1])

        # Build B matrix: B[i,j] = sum over all spins of <e_spin_i|e_spin_j>
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                # Sum dot products over all spin components
                B[i, j] = sum(
                    np.vdot(self.error_history[i][spin], self.error_history[j][spin])
                    for spin in range(self.n_fock)
                )

        # Constraint: sum of coefficients = 1
        B[n, :n] = 1.0
        B[:n, n] = 1.0
        B[n, n] = 0.0

        # Right-hand side
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0

        try:
            # Solve for coefficients
            coeffs = solve(B, rhs)
        except np.linalg.LinAlgError:
            # Fallback to most recent Fock(s) if DIIS fails
            logging.warning(
                "DIIS extrapolation failed, using most recent Fock matrices"
            )
            return self._unwrap_list(self.fock_history[-1])

        # Extrapolate Fock matrices
        fock_diis_list = []
        for spin in range(self.n_fock):
            fock_diis = np.zeros_like(self.fock_history[0][spin])
            for i in range(n):
                fock_diis += coeffs[i] * self.fock_history[i][spin]
            fock_diis_list.append(fock_diis)

        return self._unwrap_list(fock_diis_list)

    @staticmethod
    def _wrap_list(obj_or_list: FT) -> list[NDArrayf64]:
        if isinstance(obj_or_list, list):
            return obj_or_list
        return [obj_or_list]

    @staticmethod
    def _unwrap_list(list_of_obj: list[NDArrayf64]) -> FT:
        if len(list_of_obj) == 1:
            return list_of_obj[0]
        return list_of_obj
