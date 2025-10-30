#!/bin/bash
#SBATCH --job-name="mozilla_asr"
#SBATCH --nodes=2
#SBATCH --partition=base
#SBATCH --output="%x.o%j"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --mem=15G
#SBATCH --time=60:00:00

module load cuda/12.5

module load gcc/11.4.0
 
python3.11 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3.11 --version
python3.11 ./scripts/trainer.py
