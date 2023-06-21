import itertools
import os
from functools import partial
from glob import glob

from tqdm.contrib.concurrent import process_map


def parallel_glob(pattern, max_workers=os.cpu_count(), recursive=False):
    first_star_idx = pattern.index('*')
    root, pattern = pattern[:first_star_idx], pattern[first_star_idx+2:]
    sub_dirs = os.listdir(root)
    pattern_lst = [os.path.join(root, dirname, pattern)
                   for dirname in sub_dirs]
    results = process_map(partial(glob, recursive=recursive), pattern_lst,
                          max_workers=max_workers, chunksize=1)
    results = list(itertools.chain(*results))
    return results
