#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

# LATTICE VARIATION
lattice_mul = 20 #how many lattice variations to perform
lattice_variation_percent = 5

# COMPOSITION VARIATION
lattice_comp_num = 50 #how many crystals to generate (random mixing)
lattice_init = [11.7, 5.4, 11.7]

# ATOMIC VARIATION
fnum = 20 # how many atomic randomizations for a given crystal
atomic_position_variation_percent = 1


resolution = [4, 4, 6]

# possible elements
# Asite = ['Rb', 'Cs', 'K', 'Na']
# Bsite = ['Sn', 'Ge']
# Xsite = ['Cl', 'Br', 'I'] 
#test compound trend
Asite = ['Cs', 'Na']
Bsite = ['Sn']
Xsite = ['I']


#fname = 'builderOutput.csv'
#path = './bin/'    
    
