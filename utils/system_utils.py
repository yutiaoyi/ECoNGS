from errno import EEXIST
from os import makedirs, path
import os

def getTFDirs(rootPath):
    tf_dir_names = [d for d in sorted(os.listdir(rootPath)) if d.startswith("TF")]
    numTFs = len(tf_dir_names)
    return tf_dir_names, numTFs

def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
