# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import subprocess
import sys


def run(cmd):
    print(cmd)
    status, ret = subprocess.getstatusoutput(cmd)
    assert status == 0
    return ret


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python create_oss_list_objects.py oss://bucket_name/xxx/"
    path = sys.argv[1]
    assert path.startswith('oss://') and path.endswith('/'), "oss path must be a explicit dictionary"

    candidates = []
    for line in run('ossutil64 ls {} -s'.format(path)).split('\n'):
        if line.startswith(path) and not os.path.basename(line).startswith('__') and not line[-1] == '/':
            candidates.append(line)

    with open('__list_object__.txt', 'w') as f:
        for line in candidates:
            print(line, file=f)

    os.system('ossutil64 cp __list_object__.txt {}'.format(os.path.join(os.path.dirname(path), '__list_object__.txt')))

    os.system('ossutil64 set-acl -r {} public-read'.format(os.path.dirname(path)))

    os.system('rm __list_object__.txt')
