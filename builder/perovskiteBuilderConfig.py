#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""


# USE PYTHON 2.7!

fname = 'builderOutput.csv'
path = './bin/'    
    
# LATTICE OPTIONS
Asite = ['Rb', 'Cs', 'K', 'Na'] 
Bsite = ['Sn', 'Ge'] 
Xsite = ['Cl', 'Br', 'I']  
pVector = [11.4, 5.7, 11.4] # PRIMITIVE LATTICE VECTORS
latticeType = 'SimpleOrthorhombic' # DEFINED TYPE FOR VNL-ATK SCRIPT

getAll = True

# NUMBER OF CRYSTALS TO GENERATE (IN BOTH getAll = True/False CASES)
num = 10 
start = 0 # STARTING POINT IF getALL = TRUE 

# MIXING RESOLUTION: FORMAT [Asite, Bsite, Xsite]
#   RESOLUTION[i] = 1/res[i]. e.g. 2 -> R = 1/2 or 0.5  mixing
#                             e.g. 4 -> R = 1/4 or 0.25 mixing
#   Must be divisor of [8, 8, 24] for 2x2x2 Cells
#   Must be divisor of [4, 4, 12] for 2x1x2 Cells
resolution = [4, 4, 4]