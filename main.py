# General Packages
import logging
logging.basicConfig(level=logging.INFO)

# Libterachem-py system object
from terachem_util.system import System

# Local functions
from SCF import SCF

# Input
xyz = "test_systems/ferrocene.xyz"
SAD_charge_file = 'test_systems/ferrocene_charges'

# Initialize system and charges
print(f"\nReading geometry from {xyz} \n")
mol = System.from_xyzfile(xyz)

with open(SAD_charge_file, "r") as f:
    SAD_charges = [float(line.strip()) for line in f.readlines()]

print(f"Running SAD guess with charges: {SAD_charges} \n")

basis = 'sto-3g'
charge = 0
multiplicity = 1

# run SCF calculation
scf_driver = SCF(mol, basis, charge, multiplicity, SAD_charges)
scf_driver.compute('RHF')
scf_driver.compute('UHF')
scf_driver.compute('ROHF')