#!/bin/bash
cd /home/brown/wheeled-legged_RL
source /home/brown/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -u tmp/test_cod_env.py > tmp/run_log.txt 2>&1
echo "EXIT_CODE=$?" >> tmp/run_log.txt
touch tmp/run_DONE
