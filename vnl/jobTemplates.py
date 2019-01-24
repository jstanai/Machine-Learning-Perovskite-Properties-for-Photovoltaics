#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:44:21 2017

@author: Jared
"""

import csv
import textwrap 
import time

#densityMesh = '200*Hartree'
#kpoints = '(4, 4, 4)'
def getTemplateLDA(e, params, outputFile,  
                   kpoints, densityMesh):
  
    return('''
time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(''' + params['a'] + '''*Angstrom,  
                             ''' + params['b'] + '''*Angstrom,
                             ''' + params['c'] + '''*Angstrom)

# Define elements
elements = ''' + e + '''

# Define coordinates
fractional_coordinates = ''' + \
    params['fractional_coordinates'] + '''

# Set up configuration
bulk_configuration = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
    )

# -------------------------------------------------------------
# Calculator
# -------------------------------------------------------------
#----------------------------------------
# Exchange-Correlation
#----------------------------------------
exchange_correlation = LDA.PZ

numerical_accuracy_parameters = NumericalAccuracyParameters(
    occupation_method=FermiDirac(300.0*Kelvin*boltzmann_constant),
    k_point_sampling=MonkhorstPackGrid(''' + kpoints + '''),
    density_mesh_cutoff= ''' + densityMesh + ''',
    )

calculator = LCAOCalculator(
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    )

bulk_configuration.setCalculator(calculator)
nlprint(bulk_configuration)
bulk_configuration.update()
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
''')

def getTemplateGGAQuantis(e, params, outputFile,  
                   kpoints, densityMesh):
    return('''
time_ID = str(int(time.time()))

# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(''' + params['a'] + '''*Angstrom,  
                             ''' + params['b'] + '''*Angstrom,
                             ''' + params['c'] + '''*Angstrom)

# Define elements
elements = ''' + e + '''

# Define coordinates
fractional_coordinates = ''' + \
    params['fractional_coordinates'] + '''

# Set up configuration
bulk_configuration = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
    )

# -------------------------------------------------------------
# Calculator
# -------------------------------------------------------------
#----------------------------------------
# Basis Set
#----------------------------------------

basis_set = [
    GGABasis.Tin_DoubleZetaPolarized,
    GGABasis.Iodine_DoubleZetaPolarized,
    GGABasis.Germanium_DoubleZetaPolarized,
    GGABasis.Caesium_DoubleZetaPolarized,
    GGABasis.Bromine_DoubleZetaPolarized,
    GGABasis.Chlorine_DoubleZetaPolarized,
    GGABasis.Rubidium_DoubleZetaPolarized,
    GGABasis.Potassium_DoubleZetaPolarized,
    GGABasis.Sodium_DoubleZetaPolarized,
    ]

#----------------------------------------
# Exchange-Correlation
#----------------------------------------
exchange_correlation = GGA.PBES

numerical_accuracy_parameters = NumericalAccuracyParameters(
    k_point_sampling = (''' + kpoints + '''),
    density_mesh_cutoff = ''' + densityMesh + ''',
    )

calculator = LCAOCalculator(
    basis_set=basis_set,
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    )

bulk_configuration.setCalculator(calculator)
nlprint(bulk_configuration)
bulk_configuration.update()
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
''')


def getTemplatePW(e, params, outputFile,  
                   kpoints, densityMesh):
    return('''
    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(''' + params['a'] + '''*Angstrom,  
                             ''' + params['b'] + '''*Angstrom,
                             ''' + params['c'] + '''*Angstrom)

# Define elements
elements = ''' + e + '''

# Define coordinates
fractional_coordinates = ''' + \
    params['fractional_coordinates'] + '''

# Set up configuration
bulk_configuration = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
    )

# -------------------------------------------------------------
# Calculator
# -------------------------------------------------------------
#----------------------------------------
# Basis Set
#----------------------------------------

basis_set = [
    BasisGGASG15.Tin_Medium,
    BasisGGASG15.Iodine_Medium,
    BasisGGASG15.Germanium_Medium,
    BasisGGASG15.Caesium_Medium,
    BasisGGASG15.Bromine_Medium,
    BasisGGASG15.Chlorine_Medium,
    BasisGGASG15.Rubidium_Medium,
    BasisGGASG15.Potassium_Medium,
    BasisGGASG15.Sodium_Medium,
    ]

#----------------------------------------
# Exchange-Correlation
#----------------------------------------
exchange_correlation = GGA.PBE

numerical_accuracy_parameters = NumericalAccuracyParameters(
    occupation_method=FermiDirac(300.0*Kelvin*boltzmann_constant),
    k_point_sampling=MonkhorstPackGrid(''' + kpoints + '''),
    density_mesh_cutoff= ''' + densityMesh + ''',
    )


iteration_control_parameters = IterationControlParameters(
    number_of_history_steps=40,
    )

calculator = PlaneWaveCalculator(
    basis_set=basis_set,
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    iteration_control_parameters=iteration_control_parameters,
    )

bulk_configuration.setCalculator(calculator)
nlprint(bulk_configuration)
bulk_configuration.update()
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
''')

def getTemplateGGA(e, params, outputFile,  
                   kpoints, densityMesh):
    return('''
    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(''' + params['a'] + '''*Angstrom,  
                             ''' + params['b'] + '''*Angstrom,
                             ''' + params['c'] + '''*Angstrom)

# Define elements
elements = ''' + e + '''

# Define coordinates
fractional_coordinates = ''' + \
    params['fractional_coordinates'] + '''

# Set up configuration
bulk_configuration = BulkConfiguration(
    bravais_lattice=lattice,
    elements=elements,
    fractional_coordinates=fractional_coordinates
    )

# -------------------------------------------------------------
# Calculator
# -------------------------------------------------------------
#----------------------------------------
# Basis Set
#----------------------------------------

basis_set = [
    BasisGGASG15.Tin_Medium,
    BasisGGASG15.Iodine_Medium,
    BasisGGASG15.Germanium_Medium,
    BasisGGASG15.Caesium_Medium,
    BasisGGASG15.Bromine_Medium,
    BasisGGASG15.Chlorine_Medium,
    BasisGGASG15.Rubidium_Medium,
    BasisGGASG15.Potassium_Medium,
    BasisGGASG15.Sodium_Medium,
    ]

#----------------------------------------
# Exchange-Correlation
#----------------------------------------
exchange_correlation = GGA.PBE

numerical_accuracy_parameters = NumericalAccuracyParameters(
    occupation_method=FermiDirac(300.0*Kelvin*boltzmann_constant),
    k_point_sampling=MonkhorstPackGrid(''' + kpoints + '''),
    density_mesh_cutoff= ''' + densityMesh + ''',
    )


iteration_control_parameters = IterationControlParameters(
    number_of_history_steps=30,
    tolerance=5.0e-5,
    )

density_matrix_method = DiagonalizationSolver(
    processes_per_kpoint=1,
    )

algorithm_parameters = AlgorithmParameters(
    density_matrix_method=density_matrix_method,
    store_basis_on_grid=True,
    )

calculator = LCAOCalculator(
    basis_set=basis_set,
    exchange_correlation=exchange_correlation,
    numerical_accuracy_parameters=numerical_accuracy_parameters,
    iteration_control_parameters=iteration_control_parameters,
    algorithm_parameters=algorithm_parameters,
    )

bulk_configuration.setCalculator(calculator)
nlprint(bulk_configuration)
bulk_configuration.update()
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
''')

def addAnalysisFromFile(fileName):
    return('''
# -------------------------------------------------------------
# Analysis from File
# -------------------------------------------------------------
configuration = nlread(''' + '\'' + fileName + '\'' + ''' + \'.nc\', object_id='gID000')[0]
''')

def addBandStructure(bandstructure_filename):
    return('''
# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave(''' + '\'' + bandstructure_filename + '\'' + ''' + time_ID + \'.nc\', bandstructure)
''')

def addBandStructurePW(bandstructure_filename):
    return('''
# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    )
nlsave(''' + '\'' + bandstructure_filename + '\'' + ''' + time_ID + \'.nc\', bandstructure)
''')

def addOptimizeGeometryQuantis(trajectory_filename, outputFile):
    return('''         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.02*eV/Ang,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename=\'''' + trajectory_filename + '\'' + ''' + time_ID + \'.nc\',
        optimizer_method=LBFGS(),
        )
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
nlprint(bulk_configuration)
''')

def addOptimizeGeometry(trajectory_filename, outputFile):
    return('''         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename=\'''' + trajectory_filename + '\'' + ''' + time_ID + \'.nc\',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', bulk_configuration)
nlprint(bulk_configuration)
''')

def addTotalEnergy(outputFile):
    return('''
# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', total_energy)
nlprint(total_energy)
''')
    
    
def addForces(outputFile):
    return('''
# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', forces)
nlprint(forces)
''')
    
def addStress(outputFile):
    return('''
# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', stress)
nlprint(stress)
''')
    
    
def addEffectiveMass(outputFile):
    return('''
 
# -------------------------------------------------------------
# Effective Mass
# -------------------------------------------------------------
effective_mass = EffectiveMass(
    configuration=bulk_configuration,
    symmetry_label='G',
    bands_below_fermi_level=1,
    bands_above_fermi_level=1,
    stencil_order=5,
    delta=0.001*Angstrom**-1,
    direction_cartesian=[1.0, 0.0, 0.0]*Angstrom**-1,
    )
nlsave(''' + '\'' + outputFile + '\'' + ''' + time_ID + \'.nc\', effective_mass)        
    
''')









