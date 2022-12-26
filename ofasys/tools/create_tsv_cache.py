# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

# flake8: noqa: E402
import os
import pickle
import sys

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'ofasys/io/reader'))
)
from utils import line_locate


def run(cmd):
    print(cmd)
    ret = os.system(cmd)
    assert ret == 0


def local_create_cache(local_path, ofa_version=None):
    assert os.path.exists(local_path)
    if ofa_version:
        cache_path = os.path.join(os.path.dirname(local_path), os.path.basename(local_path) + '.index')
    else:
        cache_path = os.path.join(os.path.dirname(local_path), '.' + os.path.basename(local_path) + '.cache')
    print('create cache from {} to {}'.format(local_path, cache_path))
    line_pos = line_locate(local_path)
    with open(cache_path, 'wb') as f:
        if ofa_version:
            pickle.dump((len(line_pos), line_pos), f)
        else:
            np.save(f, line_pos)
    return cache_path


def oss_create_cache(oss_path):
    local_path = os.path.basename(oss_path)
    run('ossutil64 cp {} {}'.format(oss_path, local_path))
    cache_path = local_create_cache(local_path, False)
    run('ossutil64 cp {} {}/{}'.format(cache_path, os.path.dirname(oss_path), cache_path))
    run('ossutil64 set-acl {} public-read'.format(oss_path))
    run('ossutil64 set-acl {}/{} public-read'.format(os.path.dirname(oss_path), cache_path))
    run('rm -f {} {}'.format(local_path, cache_path))


# ossutil64 ls oss://ofasys/datasets/xxx/ -s | egrep "tsv$" | xargs -I {} python ofasys/tools/create_tsv_cache.py {}
if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: python create_tsv_cache.py [./a.tsv | oss://xxx/a.tsv]"
    path = sys.argv[1]
    ofa_version = '--ofa' in sys.argv

    if path.startswith('oss://'):
        if path.endswith('.tsv'):
            oss_create_cache(path)
        else:
            raise ValueError("Not supported")
    else:
        local_create_cache(path, ofa_version)
