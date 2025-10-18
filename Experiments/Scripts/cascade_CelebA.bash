#!/bin/bash
JID_JOB1=`sbatch  ../Scripts/script_run.slurm | cut -d " " -f 4`
JID_JOB2=`sbatch  --dependency=afterany:$JID_JOB1 ../Scripts/script_run.slurm | cut -d " " -f 4`
JID_JOB3=`sbatch  --dependency=afterany:$JID_JOB2 ../Scripts/script_run.slurm | cut -d " " -f 4`
JID_JOB4=`sbatch  --dependency=afterany:$JID_JOB3 ../Scripts/script_run.slurm | cut -d " " -f 4`
JID_JOB5=`sbatch  --dependency=afterany:$JID_JOB4 ../Scripts/script_run.slurm | cut -d " " -f 4`
sbatch  --dependency=afterany:$JID_JOB5 ../Scripts/script_run.slurm