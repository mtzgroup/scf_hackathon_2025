# General Packages
import logging
logging.basicConfig(level=logging.INFO)
import os
import sys
import numpy as np

# Libterachem-py system object
from systems import System

# Local functions
from SCF import SCF
from calc_charges import generate_QEq_charges

# Input
xyz = sys.argv[1]

# Initialize system and charges
print(f"\nReading geometry from {xyz} \n")
mol = System.from_xyzfile(xyz)
basis = 'cc-pvdz'
charge = 0
multiplicity = 1

# Generate partial charges for SAD guess
SAD_charges = generate_QEq_charges(xyz)
# SAD_charges = None

print(f"Running SAD guess with charges: {SAD_charges} \n")

# run SCF calculation
scf_driver = SCF(mol, basis, charge, multiplicity, SAD_charges)
scf_driver.compute('RHF')
# scf_driver.compute('UHF')
