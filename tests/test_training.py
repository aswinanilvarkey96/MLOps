import os

cwd = os.getcwd()

assert os.path.exists(str(cwd) + '/models/checkpoint.pth'), 'Checkpoint does not exit, training failed'