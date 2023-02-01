# VQE experiment

This repository contains the code used for data aquisition and processing in the article O. V. Borzenkova _et al._ "Variational simulation of Schwinger's Hamiltonian with polarization qubits", [Appl. Phys. Lett. 118, 144002 (2021)](https://doi.org/10.1063/5.0043322).

# Usage

The main script is called `experiment.py`. It accepts several options which can be listed by executing `./experiment.py -h`.

The code was used both for numerical simulations and experiments with real hardware. The option `-f` (`--fake`) switches between these two modes. Fake mode only simulates the experiment. Only `numpy` and `scipy` modules are required to run the script in this mode.

However, the real-experiment mode (no `-f` option is provided) requires some proprietary modules for communicating with devices. These modules called `thorapt` and `lbus` are not included in this repository.