# Fast approximate simulation of seismic waves using WaveNet

---

This repository reproduces the results of the paper: *[Fast approximate simulation of seismic waves with deep learning](https://arxiv.org/abs/1807.06873), NeurIPS 2018, B. Moseley, A. Markham and T. Nissen-Meyer*.

Our dataset is also provided, as well as a workshop notebook showing examples on how to run this code.

Last updated: Jan 2019

---

<img src="figures/header.png" width="600">


## Overview

- **Seismic simulation** is crucial for many geophysical applications, yet traditional approaches are **computationally expensive**.

- We use **deep learning** to simulate seismic waves.

- We show that this can offer a **fast approximate alternative** to traditional simulation methods.

- We are also able to adapt our network to carry out **fast seismic inversion**, by flipping its inputs and outputs.

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

## Getting started

A `SeismicWavenet` model can be trained very easily using the following code snippet:

```python
# define model hyperparameters
c = constants.Constants()
c["NUM_WAVE_BLOCKS"] = 1# number of WaveNet blocks to use
c["WAVE_HIDDEN_CHANNELS"] = 256# number of hidden channels in WaveNet
c["WAVE_RATES"] = [1,2,4,8,16,32,64,128,256]# dilation rates for each convolutional layer
c["WAVE_BIASES"] = False# whether to use biases in the WaveNet
c["WAVE_ACTIVATION"] = tf.nn.relu# activation function
c["CONV_FILTER_LENGTH"] = 101# filter length of the final output convolutional layer
run = main.Trainer(c)
run.train()
```

Take a look at the workshop notebook provided [here](https://github.com/benmoseley/seismic-simulation-wavenet/blob/master/Fast%20simulation%20of%20seismic%20waves%20with%20deep%20learning%20-%20Workshop%20-%20Jan%202019.ipynb) for more examples.
