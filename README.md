# Fast approximate simulation of seismic waves using WaveNet

---

Author: Ben Moseley, Centre for Autonomous Intelligent Machines and Systems, University of Oxford, bmoseley@robots.ox.ac.uk 

This repository reproduces the results of the paper: *[Fast approximate simulation of seismic waves with deep learning](https://arxiv.org/abs/1807.06873), NeurIPS 2018, B. Moseley, A. Markham and T. Nissen-Meyer*.

Our dataset is also provided; see the workshop notebook for examples on how to use this code.

Last updated: Jan 2019

---

<img src="figures/header.png" width="600">


## Overview

- **Seismic simulation** is crucial for many geophysical applications, yet traditional approaches are **computationally expensive**.

- We use **deep learning** to simulate seismic waves.

- We show that this can offer a **fast approximate alternative** to traditional simulation methods.

## Task

For this proof of principle study, we consider the simulation of **acoustic waves** propagating in synthetic **horizontally layered** media.

Specifically, we consider a single fixed point source propagating through a horizonally layered velocity model with 11 fixed receivers horizontally offset from the source, shown below.

<img src="figures/example_simulation.png" width="600"><!---include "" for proper github rendering-->

Left: input velocity model, triangles show receiver locations. Right: wavefield pressure after 1 s, using acoustic Finite-Difference (FD) modelling,  black circle shows fixed point source location.

Our task is as follows:

> Given a randomly selected layered velocity model as input, can we train a neural network to simulate the pressure response recorded at each receiver location?

We wish the network to generalise well to velocity models unseen during training.

## Workflow

We use the following workflow to complete this task;

- we preprocess the input velocity profile into its corresponding reflectivity series;

- we pass this to a deep neural network with a **WaveNet** architecture to simulate the receiver responses;

- we train this network with many example ground truth FD simulations;

- we compare the accuracy and computational performance of the trained network to FD simulation.

Our workflow is shown below.

<img src="figures/workflow.png" width="850">


## Installation

seismic-simulation-wavenet only requires Python libraries to run. We recommend setting up an new environment, for example
```bash
conda create -n workshop python=3.6  # Use Anaconda package manager
source activate workshop
```
and then installing the following dependencies:
```bash
pip install --ignore-installed --upgrade [packageURL]# install tensorflow (get packageURL from https://www.tensorflow.org/install/pip, see tensorflow website for details)
pip install tqdm requests
conda install matplotlib jupyter
```

then downloading this source code:

```bash
git clone https://github.com/benmoseley/seismic-simulation-wavenet.git
```
