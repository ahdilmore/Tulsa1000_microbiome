#!/bin/bash
#SBATCH --chdir=/projects/u19/Tulsa/birdman/medications/
#SBATCH --output=/projects/u19/Tulsa/birdman/medications/slurm_out/%x.%a.out
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --array=1-50

pwd; hostname; date

set -e

source ~/anaconda3/bin/activate birdman

echo Chunk $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX

TABLEID="medication_birdman_table"
TABLE="/projects/u19/Tulsa/final_tables/"$TABLEID".biom"
OUTDIR="/projects/u19/Tulsa/birdman/medications/inferences/antidepressants"
LOGDIR="/projects/u19/Tulsa/birdman/medications/logs/"$TABLEID
mkdir -p $OUTDIR
mkdir -p $LOGDIR

echo Starting Python script...
time python /projects/u19/Tulsa/birdman/src/antidepressants_birdman_chunked.py \
    --table-path $TABLE \
    --inference-dir $OUTDIR \
    --num-chunks $SLURM_ARRAY_TASK_MAX \
    --chunk-num $SLURM_ARRAY_TASK_ID \
    --logfile "${LOGDIR}/chunk_${SLURM_ARRAY_TASK_ID}.log" && echo Finished Python script!
