import os
import pathlib
import re


def get_checkpoint_path(checkpoint_dir: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    p = pathlib.Path(dir_path)
    new_parts = p.parts[0:-2] + ("weights",) + (checkpoint_dir,)
    final_path = pathlib.Path(*new_parts)

    if not os.path.isdir(final_path):
        final_path.mkdir(parents=True)

    return str(final_path)


checkpoint_postfix = "_"


def get_checkpoint_file(checkpoint_folder_name: str, data_name: str, index: int):
    dir = get_checkpoint_path(checkpoint_folder_name)
    return dir + "\\" + data_name + checkpoint_postfix + str(index)


def get_last_checkpoint_index(checkpoint_folder, data_names: list):
    max_indexes = [-1] * len(data_names)
    regs = [None] * len(data_names)
    for i in range(len(data_names)):
        data_name_t = data_names[i] + checkpoint_postfix
        regs[i] = re.compile(data_name_t + "([0-9]{0,}).bin$")

    dir = get_checkpoint_path(checkpoint_folder)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            for i in range(len(data_names)):
                result = regs[i].match(fname)
                if result:
                    index = int(result.group(1))
                    max_indexes[i] = max(index, max_indexes[i])
    # check for consitency
    if not all(x == max_indexes[0] for x in max_indexes):
        assert False, "Corrupted checkpoint at: {0}".format(dir)

    return max_indexes[0] if max_indexes[0] != -1 else None


def get_pretrained_path(checkpoint_name: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    p = pathlib.Path(dir_path)
    new_parts = p.parts[0:-2] + ("pretrained",) + (checkpoint_name,)
    return str(pathlib.Path(*new_parts))
