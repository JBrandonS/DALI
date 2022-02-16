#!/bin/bash
if [ $# -eq 0 ]; then
    condaenv=parthenope3.0
else
    condaenv=$1
fi
conda create -n $condaenv
source activate $condaenv || conda activate $condaenv
conda install -c conda-forge pyside2 -y
conda install numpy -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda appdirs -y
