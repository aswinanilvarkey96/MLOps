import os


def test_training():
    cwd = os.getcwd()

    assert os.path.exists(str(cwd) + '/models/checkpoint.pth'), 'Checkpoint does not exit, training failed'

if __name__ == "__main__":
    test_training()