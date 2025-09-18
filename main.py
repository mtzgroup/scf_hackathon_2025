# General Packages
import logging
logging.basicConfig(level=logging.INFO)

# Libterachem-py system object
from systems import System

# Local functions
from SCF import SCF

# Input
xyz = "test_systems/water.xyz"
SAD_charge_file = 'test_systems/water_charges'

# Initialize system and charges
print(f"\nReading geometry from {xyz} \n")
mol = System.from_xyzfile(xyz)

with open(SAD_charge_file, "r") as f:
    SAD_charges = [float(line.strip()) for line in f.readlines()]

SAD_charges = None

print(f"Running SAD guess with charges: {SAD_charges} \n")

basis = '3-21g'
charge = 0
multiplicity = 1

# run SCF calculation
scf_driver = SCF(mol, basis, charge, multiplicity, SAD_charges)
# scf_driver.compute('RHF')
scf_driver.compute('UHF')