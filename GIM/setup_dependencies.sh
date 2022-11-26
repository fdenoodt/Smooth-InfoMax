#!/usr/bin/env bash
echo "Make sure conda is installed."
echo "Installing environment:"
C:/ProgramData/Miniconda3.6/scripts/conda config --set ssl_verify false
C:/ProgramData/Miniconda3.6/scripts/conda env create -f environment.yml
source activate infomax
# C:/ProgramData/Miniconda3.6/condabin/conda.bat config --set ssl_verify false
# C:/ProgramData/Miniconda3.6/condabin/conda.bat env create -f environment.yml
# C:/ProgramData/Miniconda3.6/condabin/conda.bat activate infomax
