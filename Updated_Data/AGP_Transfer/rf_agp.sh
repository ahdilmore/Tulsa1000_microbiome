#!/bin/bash -l
#SBATCH --chdir=/panfs/adilmore/agp_comp
#SBATCH --output=/panfs/adilmore/agp_comp/rf_fi.out
#SBATCH --mail-user="adilmore@ucsd.edu"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=highmem
#SBATCH --time=7-00:00:00
source ~/anaconda3/bin/activate sklearn
python agp_rf_transfer.py
