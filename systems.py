from attrs import frozen, evolve
import numpy as np

from terachem_util.units import Bohr
from terachem_util.data import atomic_numbers
from terachem_util.typing import NDArrayf64

from functools import cached_property
from pathlib import Path
from typing import Self


@frozen
class System:
    atm_names: list[str]
    geometry: NDArrayf64  # in Bohr
    net_charge: int = 0
    multiplicity: int = 1

    @cached_property
    def charges(self) -> list[int]:
        return atomic_numbers[self.atm_names]

    @cached_property
    def n_electrons(self) -> int:
        return int(np.sum(self.charges) - self.net_charge)

    @cached_property
    def n_alpha_electrons(self) -> int:
        return (self.n_electrons + self.multiplicity - 1) // 2

    @cached_property
    def n_beta_electrons(self) -> int:
        return (self.n_electrons - self.multiplicity + 1) // 2

    @classmethod
    def from_str(cls, s: str, /, *, unit: str = "Angstrom", **kwargs) -> Self:
        fields = (line.split() for line in s.strip().splitlines())
        atmn, geom = zip(*((fd[0], [fd[1], fd[2], fd[3]]) for fd in fields))
        geom = np.array(geom, dtype=np.float64)
        if unit == "Angstrom":
            geom /= Bohr
        return cls(atm_names=list(atmn), geometry=geom, **kwargs)

    @classmethod
    def from_xyzfile(
        cls, fname: Path | str, /, *, unit: str = "Angstrom", **kwargs
    ) -> Self:
        s = Path(fname).read_text().split("\n", 2)[2]
        return cls.from_str(s, unit=unit, **kwargs)

    def with_(
        self, /, *, charge: int | None = None, spinmult: int | None = None
    ) -> Self:
        return evolve(
            self,
            **({"net_charge": charge} if charge is not None else {}),
            **({"multiplicity": spinmult} if spinmult is not None else {}),
        )
