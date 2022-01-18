#!/bin/sh
wandb login $YOUR_API_KEY
dvc pull
python -u src/models/train_model.py