# Terafit charge calculator
import torch
from terafit.qeq import StaticQEq
from terafit.qeq.params import U_P88, chi_HKK93

from utils import read_xyz

def generate_QEq_charges(xyz):

    atomic_numbers, geometry = read_xyz(xyz)

    # Calculating SAD charges
    elms = torch.from_numpy(atomic_numbers).unsqueeze(0)
    assert (chi_HKK93[elms] > 0.).all()
    geom = torch.from_numpy(geometry).unsqueeze(0)
    mask = torch.ones_like(elms, dtype=bool)

    QEq_calc = StaticQEq(electronegativity_ref=chi_HKK93,
                        U_ref=U_P88,
                        sigma_ref="from_U")
    
    charges = QEq_calc(elms, geom, mask, torch.tensor([0])).detach().numpy()

    return charges[0]