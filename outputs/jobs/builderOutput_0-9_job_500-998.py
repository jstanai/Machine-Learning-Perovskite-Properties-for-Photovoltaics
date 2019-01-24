
    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Caesium, Caesium, Caesium, Potassium, Potassium, Potassium, Potassium, Caesium, Germanium, Tin, Tin, Germanium, Germanium, Germanium, Germanium, Germanium, Iodine, Iodine, Iodine, Bromine, Iodine, Iodine, Chlorine, Chlorine, Bromine, Iodine, Bromine, Bromine, Bromine, Bromine, Chlorine, Iodine, Iodine, Chlorine, Iodine, Chlorine, Chlorine, Iodine, Iodine, Iodine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_500_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Caesium, Potassium, Caesium, Potassium, Sodium, Potassium, Sodium, Potassium, Tin, Germanium, Germanium, Germanium, Tin, Germanium, Germanium, Germanium, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine, Iodine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_501_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Sodium, Potassium, Sodium, Sodium, Sodium, Caesium, Potassium, Caesium, Tin, Tin, Germanium, Germanium, Tin, Tin, Tin, Tin, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_502_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Rubidium, Caesium, Rubidium, Rubidium, Sodium, Sodium, Rubidium, Caesium, Tin, Germanium, Germanium, Germanium, Tin, Germanium, Germanium, Germanium, Bromine, Chlorine, Iodine, Bromine, Iodine, Iodine, Iodine, Iodine, Chlorine, Chlorine, Iodine, Iodine, Iodine, Chlorine, Chlorine, Iodine, Iodine, Bromine, Bromine, Bromine, Bromine, Iodine, Iodine, Chlorine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_503_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Potassium, Sodium, Sodium, Potassium, Sodium, Potassium, Sodium, Potassium, Germanium, Tin, Tin, Germanium, Germanium, Germanium, Germanium, Germanium, Iodine, Chlorine, Iodine, Iodine, Chlorine, Chlorine, Iodine, Bromine, Iodine, Iodine, Chlorine, Chlorine, Bromine, Iodine, Chlorine, Iodine, Iodine, Bromine, Bromine, Iodine, Bromine, Iodine, Bromine, Iodine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_504_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Caesium, Caesium, Caesium, Caesium, Caesium, Caesium, Caesium, Caesium, Tin, Tin, Germanium, Germanium, Germanium, Germanium, Tin, Tin, Bromine, Chlorine, Chlorine, Iodine, Iodine, Iodine, Bromine, Bromine, Bromine, Bromine, Chlorine, Bromine, Bromine, Iodine, Chlorine, Chlorine, Chlorine, Iodine, Bromine, Bromine, Bromine, Bromine, Bromine, Iodine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_505_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Rubidium, Rubidium, Rubidium, Rubidium, Rubidium, Rubidium, Rubidium, Rubidium, Germanium, Tin, Tin, Germanium, Germanium, Germanium, Germanium, Germanium, Bromine, Chlorine, Chlorine, Iodine, Bromine, Iodine, Chlorine, Chlorine, Bromine, Bromine, Bromine, Bromine, Bromine, Bromine, Bromine, Chlorine, Iodine, Chlorine, Iodine, Bromine, Bromine, Iodine, Iodine, Bromine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_506_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Rubidium, Sodium, Potassium, Sodium, Sodium, Rubidium, Potassium, Sodium, Germanium, Germanium, Germanium, Germanium, Germanium, Germanium, Germanium, Germanium, Chlorine, Bromine, Iodine, Chlorine, Iodine, Iodine, Chlorine, Chlorine, Chlorine, Bromine, Iodine, Bromine, Iodine, Iodine, Bromine, Iodine, Chlorine, Iodine, Iodine, Bromine, Iodine, Iodine, Iodine, Bromine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_507_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Potassium, Rubidium, Caesium, Caesium, Caesium, Caesium, Potassium, Rubidium, Germanium, Tin, Germanium, Germanium, Tin, Germanium, Germanium, Germanium, Iodine, Iodine, Iodine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Iodine, Chlorine, Chlorine, Iodine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Chlorine, Iodine, Chlorine, Chlorine, Chlorine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_508_' + time_ID + '.nc', effective_mass)        
    

    

time_ID = str(int(time.time()))
# -------------------------------------------------------------
# Bulk Configuration
# -------------------------------------------------------------

# Set up lattice
lattice = SimpleOrthorhombic(11.4*Angstrom,  
                             5.7*Angstrom,
                             11.4*Angstrom)

# Define elements
elements = [Rubidium, Rubidium, Rubidium, Caesium, Caesium, Rubidium, Rubidium, Rubidium, Tin, Tin, Germanium, Germanium, Tin, Tin, Tin, Tin, Bromine, Bromine, Bromine, Chlorine, Bromine, Iodine, Chlorine, Iodine, Bromine, Chlorine, Iodine, Bromine, Iodine, Iodine, Chlorine, Bromine, Bromine, Bromine, Bromine, Bromine, Iodine, Chlorine, Bromine, Chlorine]

# Define coordinates
fractional_coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.25, 0.5, 0.25], [0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.75, 0.5, 0.75], [0.25, 0.5, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.0], [0.75, 0.5, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75], [0.25, 0.0, 0.25], [0.25, 0.0, 0.75], [0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]

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
    k_point_sampling=MonkhorstPackGrid(na=6, nb=12, nc=6,),
    density_mesh_cutoff= 200*Hartree,
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', bulk_configuration)
         
# -------------------------------------------------------------
# Optimize Geometry
# -------------------------------------------------------------
bulk_configuration = OptimizeGeometry(
        bulk_configuration,
        max_forces=0.01*eV/Ang,
        max_stress=0.01*GPa,
        max_steps=200,
        max_step_length=0.3*Ang, 
        trajectory_filename='./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc',
        optimizer_method=LBFGS(),
        constrain_bravais_lattice=True,
        )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', bulk_configuration)
nlprint(bulk_configuration)

# -------------------------------------------------------------
# Total Energy
# -------------------------------------------------------------           
total_energy = TotalEnergy(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', total_energy)
nlprint(total_energy)

# -------------------------------------------------------------
# Forces
# -------------------------------------------------------------           
forces = Forces(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', forces)
nlprint(forces)

# -------------------------------------------------------------
# Stress
# -------------------------------------------------------------           
stress = Stress(bulk_configuration)
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', stress)
nlprint(stress)

# -------------------------------------------------------------
# Bandstructure
# -------------------------------------------------------------
bandstructure = Bandstructure(
    configuration=bulk_configuration,
    route=['Y', 'G', 'X', 'S', 'R', 'U', 'X', 'S', 'Y', 'T', 'Z', 'G', 'U', 'Z'],
    points_per_segment=20,
    bands_above_fermi_level=All
    )
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', bandstructure)

 
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
nlsave('./outputs/jobs/builderOutput_0-9_job_complete_509_' + time_ID + '.nc', effective_mass)        
    
