"""
A simple SCF driver class
"""

# imports
from systems import System
from SAD import generate_SAD_guess_partial_charge
from RHF import RHF
from UHF import UHF
from terachem_util import use, gpubox

class SCF:
    """
    Hartree-Fock SCF calculation wrapper
    """

    def __init__(self, mol, basis, charge=0, multiplicity=1, SAD_charges=None):
        """
        Initialize SCF driver

        Parameters:
        -----------
        mol : System or str
            Molecular system (System object or geometry string)
        basis : str
            Basis set name
        charge : int
            Molecular charge
        multiplicity : int
            Spin multiplicity (2S + 1)
        SAD_charges : lst or None
            List of (partial) charges to be used for the SAD initial guess generations
        """

        # Handle molecular system
        if isinstance(mol, str):
            self.mol = System.from_str(mol)
        else:
            self.mol = mol

        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity

        if SAD_charges is None:
            self.SAD_charges = [0 for atom in range(len(mol.atm_names))]
        else:
            self.SAD_charges = SAD_charges

    def compute(self,scf_type,conv_acc='DIIS'):
        """
        Performs SCF calculation of specified type

        Parameters:
        -----------
        scf_type : str
            Type of SCF calculation: 'RHF' or 'UHF'
        conv_acc : str
            Type of SCF convergence scheme 'DIIS' or 'SOSCF'
        """

        with use(gpubox([0])):
            # Generate initial guess from a (charged) SAD
            D0 = generate_SAD_guess_partial_charge(self.mol.atm_names,self.basis,self.SAD_charges)

            # Run actual SCF calculation
            if scf_type == 'RHF':
                print('Running Restricted Hartree-Fock\n')
                rhf_driver = RHF(self.mol,self.basis)
                rhf_driver.kernel(dm0=D0)
            elif scf_type == 'UHF':
                print('Running Unrestricted Hartree-Fock\n')
                uhf_driver = UHF(self.mol,self.basis)
                uhf_driver.kernel(dm0=D0)
            else:
                print('What the fuck are you doing? Use RHF or UHF\n')


        return 0
