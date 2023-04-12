import os
import pathlib


def get_checkpoint_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    p = pathlib.Path(dir_path)
    new_parts = p.parts[0:-2] + ("pretrained",)
    return str(pathlib.Path(*new_parts)) + "\\"
