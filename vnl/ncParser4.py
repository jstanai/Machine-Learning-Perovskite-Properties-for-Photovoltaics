#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Thu Dec  7 15:35:32 2017


@author: Jared

PURPOSE: To parse *.nc files and create a data set for machine learning
"""



import os
import csv
import sys
import numpy as np

def main():
 
	   
	#fileFolder = sys.argv[1] # 'C:\Users\ga85pag\TestData' dont use quote on input)
	fileFolder = '/mnt/local/home/ga85pag/.vnl/perovskiteProject/data/18062018_benchmark/run05/'

	outputCSV = 'run05_ncParser4.csv'
	
	#
	#
	# Parse 'COMPLETE.NC' files (configurations)
	i = 0
	for filename in os.listdir(fileFolder):
		if filename.endswith(".nc") and 'complete' in filename: i += 1
	
   	output = [[0] for j in range(i)]

	i = 0
	for filename in os.listdir(fileFolder):
       
		if filename.endswith(".nc") and 'complete' in filename: 
           
			print(filename.split('_')[-1].split('.')[0])
			
			# Configuration Data
			crystal_id = filename.split('_')[-1].split('.')[0]
			config = nlread(fileFolder + filename, BulkConfiguration)
			# gID mapping here from quantumwise outputs
			config_start = config[0]
			config_end = config[1]
			
			q = config_start.bravaisLattice().primitiveVectors()  
			q = q.inUnitsOf(Ang)
			q = q.flatten()
			primVectors_start = np.ndarray.tolist(q)

			q = config_end.bravaisLattice().primitiveVectors() 
			q = q.inUnitsOf(Ang)
			q = q.flatten()
			primVectors_end = np.ndarray.tolist(q)

			frac_coords = config_end.fractionalCoordinates()
	
			frac_coords = np.ndarray.tolist(frac_coords)

			elements = [x.name() for x in config_end.elements()]
			
			# Experimental Settings Data
			calculator = config_end.calculator()
			
			density_mesh_cutoff = calculator.numericalAccuracyParameters().densityMeshCutoff()
			density_cutoff = calculator.numericalAccuracyParameters().densityCutoff()
			electron_temperature = calculator.numericalAccuracyParameters().occupationMethod().broadening()
			ka = calculator.numericalAccuracyParameters().kPointSampling().na()
			kb = calculator.numericalAccuracyParameters().kPointSampling().nb()
			kc = calculator.numericalAccuracyParameters().kPointSampling().nc()
			pseudo_basis = 'SG15_Medium'
			XC = 'GGA.PBE'
			spin = calculator.exchangeCorrelation().spinType()

			# Add read log file command to find OPT for geometry optimization params
			
			# bandgaps
			bandstructure = nlread(fileFolder + filename, Bandstructure)[0]
			indirectGap = bandstructure._indirectBandGap().inUnitsOf(eV)
			directGap = bandstructure._directBandGap().inUnitsOf(eV)						
			
			
			#nlprint(fileFolder + 'bandstructure' + filename, Bandstructure)
			
			# Determine indirect band gap
			energies = bandstructure.evaluate().inUnitsOf(eV)

			e_valence_max = -1.e10
    			e_conduction_min = 1.e10
    			e_gap_min = 1.e10
    			i_valence_max=0
    			i_conduction_min=0
    			i_gap_min = 0
    			n_valence_max=0
    			n_conduction_min=0
    			n_gap_min = 0

			# Locate extrema
    			for ii in range(energies.shape[0]):
        			# find first state below Fermi level
    				n=0
    				while n < energies.shape[1] and energies[ii][n] < 0.0:
       					n=n+1

        			#find maximum of valence band
        			if (energies[ii][n-1] > e_valence_max):
            				e_valence_max=energies[ii][n-1]
            				i_valence_max=ii
            				n_valence_max=n-1
        			#find minimum of conduction band
        			if (energies[ii][n] < e_conduction_min):
           				e_conduction_min=energies[ii][n]
            				i_conduction_min=ii
            				n_conduction_min=n
        			#find minimum band band
        			if (energies[ii][n]-energies[ii][n-1] < e_gap_min):
            				e_gap_min=energies[ii][n]-energies[ii][n-1]
            				i_gap_min=ii
            				n_gap_min = n-1
      

			# Print out results
			print('Valence band maximum (eV) ',e_valence_max, 'at ')
			
			val_max = e_valence_max
			kpoint_val_max = bandstructure.kpoints()[i_valence_max].tolist()
			print(kpoint_val_max)
			print('Conduction band minimum (eV)',e_conduction_min, 'at ')
			
			kpoint_con_min = bandstructure.kpoints()[i_conduction_min].tolist()
			con_min = e_conduction_min
			print(kpoint_con_min)


				
			# Energy total
			energy = nlread(fileFolder + filename, TotalEnergy)[0]
			components = energy.components()			
			totalEnergy = energy.evaluate().inUnitsOf(eV)
			#totalEnergy = totalEnergy.flatten()
			#totalEnergy = np.ndarray.tolist(totalEnergy)
			
			# Effective Mass
			em = nlread(fileFolder + filename, EffectiveMass)[0]
			em = list(em.evaluate())
			em_hole = em[0][0]
			em_electron = em[1][0]
			
			output[i] = [crystal_id, 
						 primVectors_start, primVectors_end, elements, frac_coords,
						 indirectGap, directGap, 
						 components, totalEnergy, em_hole, em_electron, 
						 density_mesh_cutoff, density_cutoff, electron_temperature, ka, kb, kc, 
						 pseudo_basis, XC, spin, 
						 val_max, kpoint_val_max, con_min, kpoint_con_min]
			#print(output[i])
			i += 1	
					      
		else:
            
			continue

	header = ['crystal_id',
			  'cellPrimVectors_start', 'cellPrimVectors_end', 'elements', 'fractional_Coordinates',
			  'indirectGap', 'directGap', 
			  'energy_components', 'totalEnergy', 'effectiveMass_hole', 'effectiveMass_electron',
			  'densityMeshCutoff', 'densityCutoff', 'electronTemperature', 'ka', 'kb', 'kc', 
			  'pseudo_basis', 'XC', 'spin', 
			  'valenceMax_eV', 'kpoint_valenceMax', 'conductionMin_eV', 'kpoint_conductionMin']
	
	with open(fileFolder + outputCSV, 'w') as f:
		writer = csv.writer(f, dialect=csv.excel)
		writer.writerow(header)
		for i in output:
			writer.writerow(i)
	

		
	
if __name__ == '__main__':
    
	main()