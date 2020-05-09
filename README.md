# Machine Learning Perovskites

This software was developed during my masters research and enables the user to perform end-to-end analysis of VNL-ATK 2017 Density Functional Theory (DFT) calculations using Kernel Ridge Regression. This work was published and can be cited as:

Stanley, J.C., Mayr, F. and Gagliardi, A. (2020), Machine Learning Stability and Bandgaps of Lead‐Free Perovskites for Photovoltaics. Adv. Theory Simul., 3: 1900178. doi:10.1002/adts.201900178

This software contains several important modules for building crystals, running predictions, and automating VNL job script creation. Additionally, it contains over 300 quantum calculations for perovskite compositional mixtures developed using VNL-ATK 2017. We use these data to train a machine learning algorithm, using novel property density distribution function (PDDF) to encode the local atomic environment based on a fundamental set of atomic properities. This allows us to find a general algorithm to prediction of key material properites, such as bandgap and formation energy. 

![Results](https://github.com/jstanai/Machine-Learning-Perovskite-Properties-for-Photovoltaics/blob/master/images/Results.png "Key Results")

This software contains several modules to help researches build perovskite crystals from scratch, visualize them
There are three primary modules used to accomplish this task, which can also be used separately for various kinds of analysis and job-scripting. Please feel free to contact me with questions!

Jared Stanley
https://www.linkedin.com/feed/
jcamstan@gmail.com


## Table of Contents
[Introduction](#introduction)

[Data](#data)

[Builder Module](#builder-module)

[VNL Module](#vnl-module)

[ML Module](#ml-module)

[Acknowledgements](#acknowledgements)


### Introduction

Using a database of crystals generated in VNL-ATK 2017, we can apply supervised machine learning to predict important material properties for many kinds of applications, such as photovoltaics. Quantities such as the direct or indirect bandgap, formation energy, convex-hull energy distance, effective masses m<sub>e</sub> or m<sub>h</sub>, and many other quantities from the VNL framework can be set as the response, and then predicted using design matrix **X**.

Here, we use a Kernel Ridge Regression as a dual representation of the the regularized least-squares formalism with the Gaussian kernel

![Kernel](hhttps://github.com/jstanai/Machine-Learning-Perovskite-Properties-for-Photovoltaics/blob/master/images/GaussianKernel.png "Gaussian Kernel")

as a similarity measure between two crystal feature vectors. Technically, implicit feature maps are defined by the kernel function, whereas *descriptors* generated from the material fingerprint would correspond to each observation **x**. Such descriptors are encoded from elemental properties and their configuration within each crystal using the PDDF method. 

More details can be found in the attached PDF of the full thesis.

### Data
#### Validation of Density Functional Theory Parameters

Extensive convergence tests were performed on 30 mixed and non-mixed compositions from
the same chemical space to validate the chosen DFT parameters. We started with testing
density mesh cutoff energies of Edmc = 40, 70, 100, and 150 Hartree, and k-point grids of
k<sub>a</sub> × k<sub>b</sub> × k<sub>c</sub> = 3×6×3, 6×12×6, 10×20×10, and 12×24×12 at a variety of stress and force
tolerances. While most non-mixed compositions converged rapidly under relaxed geometry optimization parameters with a cutoff energy around 100 Hartree and k-point mesh of
6x12x6, we found that much stricter tolerances were required for mixed compositions.
An example for the required cutoff energy and kb values for two such compounds are shown
in Figure 1. The right-upward-pointing triangles are computed from lower-fidelity DFT
parameters, with relaxed tolerances of 0.1 GPa, 20 meV/˚A , and 10<sup>−4</sup>
for the stress, force
and SCF tolerance respectively. Taking the reference point as (kmesh, Edmc) = (6×12×6, 250

Hartree) with the stricter force, stress and SCF tolerances of 10 meV/˚A, 0.01 GPa, and 5 ×
10<sup>−5</sup> used in this study, our choice of (kmesh, Edmc) = (6×12×6, 200 Hartree) has an absolute
bandgap error of 11 meV and 150 meV for K<sub>2</sub>CsNaSn<sub>3</sub>GeCl<sub>9</sub>I<sub>3</sub> (a) and K<sub>2</sub>RbNaSn<sub>3</sub>GeCl<sub>8</sub>Br<sub>4</sub>
(b) respectively. At the more relaxed tolerances, this increases to 810 meV and 239 meV.
Using the reference (kmesh, Edmc) = (16×32×16, 200 Hartree) the errors for each compound
are 18 meV and 48 meV (804 meV and 42 meV) for the high-fidelity (low-fidelity) tolerance
parameters. Convergence errors add to the irreducible error, limiting prediction power of
our algorithm.

![Convergence](https://github.com/jstanai/Machine-Learning-Perovskite-Properties-for-Photovoltaics/blob/master/images/Convergence.png "Convergence Testing")

#### Data Sets

Following the paper, this repository has several data sets contained in the `/data` directory. 

- crystals: 
  - benchmark and convergence data in `/benchmark`
  - CIF files for visualization in `/cif_files`
  - 344 original VNL outputs in `/d2`
  - 8 Pb calculations in `/lead`
- features:
  - transformed benchmark and convergence data in `/benchmark`
  - 344 transformed crystals in `/d2_paper`
  - dummy crystals used for larger trend analysis
- hull:
  - Convex hull calculations using the qmdb database

Many of the data can be found within the paper and supporting information. In particular, Pb data, thermodynamic data, and convergence data can be found in the Supporting Information section. 

### Builder Module

#### File definitions
- `perovskiteConfig.py` - configures elements, lattice vectors, outputs, etc. for perovskiteBuilder.py
- `perovskiteBuilder.py` - generates csv of perovskite crystal structures for ingestion into `vnl` module
- `dummyCrystalConfig.py` - configures parameters for dummy crystal construction
- `dummyCrystalBuilder.py` - builds dummy crystal definitions automatically

## VNL Module

This module will take csv inputs of crystals, and generate python scripts for VNL jobs automatically. The goal is to 'fill-in' standard templates with this information so they can be sent to a cluster and their results parsed. 

#### File definitions
- `jobTemplates.py` - contains job templates that are useful starting points for many VNL jobs. 
- `jobScripter.py` - fills-in jobs and binds them together for parallel and fault-tolerance execution.
- `ncParser4.py` - parses job results to extract relevant crystal information.

## ML Module

This module contains all the supporting scripts to perform machine learning on data sets generated in the previous `vnl` module. There are several helper functions and dictionary definitions which are used to evaluate and train on the data. The scripts have been designed to allow easy feature exploration and automated interaction-term checks, PDDF computation, and KRR hyper-parameter tuning. 

#### File definitions
- `elements.py` - element-specific data, property label and mixture definitions, long/short symbol conversion.
- `errors.py` - simple error helpers.
- `feature.py` - PDDF feature generation algorithm (highly optimized for speed).
- `helpers.py` - simple helper functions.
- `integrator.py` - fast integration functionality to support `feature.py`.
- `krr.py` - kernel ridge regression defintions and hyper-parameters search functions.
- `preprocessor.py` - feature scalers and transformers.

## Acknowledgements
I want to thank my advisor Prof. Dr. Alessio Gagliardi for his insight, passion, and
mentoring throughout this thesis, as well as the incredible opportunities he has
provided me to develop myself both personally and professionally. I am also
grateful for the support of my colleagues in both the Physics and Electrical and
Computer Engineering departments, the help of Prof. Dr. Aliaksandr Bandarenka,
and the resources provided by the Technische Universität München that have
made this project possible. 



