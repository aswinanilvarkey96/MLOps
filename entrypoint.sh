#!/bin/sh
echo 'wandb'
wandb login $YOUR_API_KEY
echo 'dvc pull'
dvc pull
echo 'dvc pull done'
python -u src/models/train_model.py
echo 'docker done'