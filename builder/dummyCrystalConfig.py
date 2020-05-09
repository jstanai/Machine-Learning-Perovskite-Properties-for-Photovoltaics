#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""

# LATTICE VARIATION
lattice_mul = 20 #how many lattice variations to perform
lattice_variation_percent = 10
keep_aspect = False

# COMPOSITION VARIATION
# upper limit to sim num
lattice_comp_num = 500 #how many crystals to generate (random mixing)
lattice_init = [11.7, 5.4, 11.7]
#lattice_init = [12.5, 6.19, 12.5]

# ATOMIC VARIATION
fnum = 20 # how many atomic randomizations for a given crystal
atomic_position_variation_percent = 3

resolution = [1, 1, 2] #cannot change to [4, 4, 12] due to itertools overflow

# possible elements
# Asite = ['Rb', 'Cs', 'K', 'Na']
# Bsite = ['Sn', 'Ge']
# Xsite = ['Cl', 'Br', 'I'] 
#test compound trend
Asite = ['Cs', 'Rb', 'K', 'Na']
Bsite = ['Sn', 'Ge']
Xsite = ['I', 'Cl', 'Br']


#fname = 'builderOutput.csv'
#path = './bin/'    
    
