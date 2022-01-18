#!/bin/sh
echo 'wandb'
wandb login $SECRET_KEY
echo 'dvc pull'
dvc pull
echo 'dvc pull done'
python -u src/models/train_model_arg.py
echo 'docker done'