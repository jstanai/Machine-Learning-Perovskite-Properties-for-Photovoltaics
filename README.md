# Machine Learning Perovskites

This software enables the user to perform end-to-end analysis of VNL-ATK 2017 Density Functional Theory (DFT) calculations using machine learning. There are three primary modules used to accomplish this task, which can also be used separately for various kinds of analysis and job-scripting. 


### Table of Contents
[Introduction](#introduction)
[Builder Module](#builder-module)
[VNL Module](#vnl-module)
[ML Module](#ml-module)
[Example Usage](#example-usage)
[Acknowledgements](#acknowledgements)


## Introduction

Using a database of crystals generated in VNL-ATK 2017 [***], we can apply supervised machine learning to predict important material properties for many kinds of applications, such as photovoltaics. Quantities such as the direct or indirect bandgap $E_{g}$, formation energy $\Delta H_{f}$, convex-hull energy $\Delta H_{hull}$, effective masses $m_{e}$ or $m_{h}$, and many other quantities from the VNL framework can be set as the target $\hat{y}$, and then predicted using a design matrix $\bf{X}$ of dimension $(n, m)$ for $n$ observations of $m$ features.

Here, we use a Kernel Ridge Regression as a dual representation of the the regularized least-squares formalism with the Gaussian kernel $k(\vec{x_{1}}, \vec{x_{2}}) = \exp(-\gamma \| \vec{x_{1}} - \vec{x_{2}}\|_{2}^{2} )$ as a similarity measure between two crystal feature vectors $\vec{x}$. Technically, implicit feature maps are defined by the kernel function, whereas *descriptors* generated from the material fingerprint would correspond to each observation $\vec{x}$. Such descriptors are encoded from elemental properties and their configuration within each crystal using a density function as detailed in [***].



1) builder Module
2) vnl Module
3) ml Module

## Builder Module

### File definitions
- perovskiteBuilder.py - generates csv of perovskite crystal structures
- perovskiteConfig.py - configures elements, lattice vectors, outputs, etc. for perovskiteBuilder.py
- dummyCrystalBuilder.py - builds dummy crystals (see Example Usage)


## VNL Module

### File definitions

## ML Module

### File definitions

## Example Usage


## Acknowledgements


