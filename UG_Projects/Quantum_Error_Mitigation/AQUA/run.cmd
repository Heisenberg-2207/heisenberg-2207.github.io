#!/bin/bash
#PBS -o script_output.log
#PBS -e script_error.err
#PBS -l walltime=14:00:00
#PBS -l select=1:ncpus=1

# Print initial environment for debugging
echo "Initial PATH: $PATH"
echo "Initial PYTHONPATH: $PYTHONPATH"

# Explicitly set Miniconda path
export PATH="$HOME/miniconda3/bin:$PATH"

# Conda Setup
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py311

# Verify Python version after activation
which python
python --version

# Temp Workspace Setup
tpdir=$(echo $PBS_JOBID | cut -f 1 -d .)
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir || exit

# File Transfer
cp -R $PBS_O_WORKDIR/* .

# Python Execution
echo "=== PYTHON ENVIRONMENT ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

python script.py > output.log 2>&1

# Results Cleanup
cp output.log $PBS_O_WORKDIR/
[ -f *.result ] && cp *.result $PBS_O_WORKDIR/
rm -rf $tempdir

echo "=== SERIAL JOB COMPLETED ==="
