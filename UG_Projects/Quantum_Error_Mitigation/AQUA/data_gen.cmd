#!/bin/bash
#PBS -o logfile.log
#PBS -e errorfile.err
#PBS -l walltime=14:00:00
#PBS -l select=1:ncpus=1:ngpus=0

# Load Conda and activate environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py311

# Create temporary directory using job ID
tpdir=$(echo $PBS_JOBID | cut -f 1 -d .)
tempdir=$HOME/scratch/job$tpdir
mkdir -p "$tempdir"
cd "$tempdir" || exit

# Copy files from original directory
cp -R "$PBS_O_WORKDIR"/* .

# Run Python with virtual environment
python test.py 2>&1 | tee simulation_output.log

# Cleanup and move results
mv "$tempdir" "$PBS_O_WORKDIR"/
