# ME5406 Project 1

This repo contains my project to compare the performance acress Monte Carlo Without Exploring Starts, SARSA and Q-Learning for FrozenLake-V1.

This repo has been been tested for python 3.6.10 and 3.9.12. 

# Prerequisites
This repo was built under a conda environment. Make sure to install `conda` in you Operating System. Here are the instructions to install conda in [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) and [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

This is to ensure that this repo met the requirements of the project: to run on python 3.6.

# Windows


1. Clone this Repository. Or Unzip the folder containing this repository. And open a terminal inside the folder. 

2. `conda env create -f environment_windows.yml`

    This will create the virtual environment.

3. `conda activate me5406_1`

    Or what the result of step to tells you. 

4. Once inside the Environment, you can run the individual python Files in the terminal using 

    `python <file-name>`

- ES_4x4.py - Monte Carlo without Exploring Start 4x4 Grid
- ES_10x10.py - Monte Carlo without Exploring Start 10x10 Grid
- SARSA_4x4.py - SARSA 4x4 Grid
- SARSA_10x10.py - SARSA 10x10 Grid
- QL_4x4.py - Q-Learning 4x4 Grid
- QL_10x10.py - Q-Learning 10x10 Grid


# Linux

1. Clone this Repository. Or Unzip the folder containing this repository. And open a terminal inside the folder. 

2. `conda env create -f environment_ubuntu.yml`

    This will create the virtual environment.

Steps 3 and 4 are similar to windows once inside the conda environment. 


# Output

The output of the python files are the success rate during evaluation in the terminal and a graph window depicting the frequency successful training and evaluation episodes and the number of steps taken to reach the goal. 