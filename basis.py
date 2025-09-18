from pathlib import Path
from os import environ
from itertools import islice, chain
from typing import Self, Protocol, Callable

from attrs import define
import numpy as np

from terachem_util.typing import NDArrayf64, NDArrayInt

BASIS_PATH = Path(environ["TeraChem"]) / "basis"

# Angular momentum mapping
MAX_ANGMOM = 5  # Maximum supported angular momentum
ANGL_INT_TO_LETTER = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
ANGL_LETTER_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def parse_atom(fh):
    basis = []
    while li := next(fh, "").strip():
        orb, n = li.split()
        bx = np.loadtxt(islice(fh, int(n)), dtype=float, unpack=True)  # alpha, coeff
        bx = np.copy(bx, order="C").reshape(2, -1)
        basis.append((ANGL_LETTER_TO_INT[orb.lower()], bx))
    return basis


def parse_basis_file(fh):
    all_basis = {}
    for li in fh:
        if li.startswith("ATOM"):
            all_basis[li.strip().split()[1]] = parse_atom(fh)
    return all_basis


def normalize_shell(angl, alpha, coeff):
    c1 = coeff * np.power(2 * alpha / np.pi, 0.75) * 2**angl * np.power(alpha, angl / 2)
    # fmt: off
    norm = (
        c1[:, None] @ c1[None, :]
        / np.power(alpha[:, None] + alpha[None, :], angl + 1.5)
    ).sum()
    # fmt: on
    norm *= 1 / 2**angl * np.power(np.pi, 1.5)
    norm = 1 / np.sqrt(norm)
    c1 *= norm
    return c1


class BasisConverter(Protocol):
    spherical_to_cartesian: Callable[[NDArrayf64, NDArrayf64], None]
    cartesian_to_spherical: Callable[[NDArrayf64, NDArrayf64], None]


@define
class BasisInfo:
    nao_cart: int
    nao_sph: int
    to_angmom_sph: NDArrayInt
    to_atom_sph: NDArrayInt
    to_angmom_cart: NDArrayInt
    to_atom_cart: NDArrayInt
    ranges_atom_cart: list[list[NDArrayInt]]
    shells_by_angmom: list[list[tuple[int, NDArrayf64, NDArrayf64]]]

    def spherical_to_cartesian_order2(
        self, intbox: BasisConverter, p: NDArrayf64
    ) -> NDArrayf64:
        """
        Transform a order-2 tensor from spherical basis (atom-major)
        to cartesian basis (angular momentum-major).
        """
        # Reorder to angular momentum-major
        p_angmom = p[self.to_angmom_sph, :][:, self.to_angmom_sph]

        # Convert to cartesian using intbox
        p_cart = np.zeros((self.nao_cart, self.nao_cart), dtype=np.float64)
        intbox.spherical_to_cartesian(p_angmom, p_cart)

        return p_cart

    def cartesian_to_spherical_order2(
        self, intbox: BasisConverter, p: NDArrayf64
    ) -> NDArrayf64:
        """
        Transform a order-2 tensor from cartesian basis (angular momentum-major)
        to spherical basis (atom-major).
        """
        # Convert to spherical using intbox
        p_sph = np.zeros((self.nao_sph, self.nao_sph), dtype=np.float64)
        intbox.cartesian_to_spherical(p, p_sph)

        # Reorder back to atom-major for other programs
        p_atom = p_sph[self.to_atom_sph, :][:, self.to_atom_sph]

        return p_atom

    @classmethod
    def from_basis_file(cls, basis_name: str, atm_names: list[str]) -> Self:
        with open(BASIS_PATH / basis_name, "r") as fh:
            basis = parse_basis_file(fh)

        # Group shells by angular momentum
        shells_by_angmom = [[] for _ in range(MAX_ANGMOM + 1)]
        for i, atm in enumerate(atm_names):
            shells = basis[atm]
            for angl, (alpha, coeff) in shells:
                coeff = normalize_shell(angl, alpha, coeff)
                shells_by_angmom[angl].append((i, alpha, coeff))

        # Ranges for each atoms
        ranges_sph = [[] for _ in range(len(atm_names))]
        ranges_cart = [[] for _ in range(len(atm_names))]

        nao_cart = 0
        nao_sph = 0
        for angl, shells in enumerate(shells_by_angmom):
            snao_cart = (angl + 1) * (angl + 2) // 2  # Cartesian
            snao_sph = 2 * angl + 1  # Spherical
            for i, alpha, coeff in shells:
                ranges_sph[i].append(np.arange(nao_sph, nao_sph + snao_sph))
                ranges_cart[i].append(np.arange(nao_cart, nao_cart + snao_cart))
                nao_sph += snao_sph
                nao_cart += snao_cart

        # Compute reordering indices
        def _slices_to_perm(ranges: list[list[NDArrayInt]]):
            to_atom = np.concatenate(list(chain.from_iterable(ranges)))
            to_angmom = np.empty_like(to_atom)
            to_angmom[to_atom] = np.arange(to_atom.size)
            return to_angmom, to_atom

        to_angmom_sph, to_atom_sph = _slices_to_perm(ranges_sph)
        to_angmom_cart, to_atom_cart = _slices_to_perm(ranges_cart)

        bi = cls(
            nao_cart=nao_cart,
            nao_sph=nao_sph,
            to_angmom_sph=to_angmom_sph,
            to_atom_sph=to_atom_sph,
            to_angmom_cart=to_angmom_cart,
            to_atom_cart=to_atom_cart,
            ranges_atom_cart=ranges_cart,
            shells_by_angmom=shells_by_angmom,
        )
        return bi


class ShellInitializer(Protocol):
    init_shell: Callable[[int, int, int, NDArrayf64, NDArrayf64], None]


def populate_basis_shells(
    box: ShellInitializer, basis_name: str, atm_names: list[str]
) -> BasisInfo:
    """Prepare intbox with the given basis set and atoms."""
    bi = BasisInfo.from_basis_file(basis_name, atm_names)
    initialize_integral_shells(box, bi.shells_by_angmom)
    return bi


def initialize_integral_shells(
    box: ShellInitializer,
    shells_by_angmom: list[list[tuple[int, NDArrayf64, NDArrayf64]]],
):
    init_shell = box.init_shell
    for angl, shells in enumerate(shells_by_angmom):
        for i, alpha, coeff in shells:
            init_shell(angl, i, len(alpha), alpha, coeff)
