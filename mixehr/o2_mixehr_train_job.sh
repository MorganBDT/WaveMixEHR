#!/bin/bash
#SBATCH -c 8                               # Request one core
#SBATCH -t 0-36:00                         # Runtime in D-HH:MM format
#SBATCH --mem=128000M                         # Memory total in MB (for all cores)
#SBATCH -p medium                             # Partition to run in (e.g. short, gpu)
#SBATCH -o ./o2_results/o2_results_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./o2_results/o2_errors_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=FAIL

module load conda2/4.2.13

source activate ml1

./mixehr -f ./mixmimic/$ehrdata -m ./mixmimic/$ehrmeta -k 75 -i 500 \
	--inferenceMethod JCVB0 --maxcores 8 \
	--outputIntermediates 