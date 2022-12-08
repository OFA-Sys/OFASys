# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import importlib
import os

OFA_LOGGING_AUTO_IMPORT = os.environ.get('OFA_LOGGING_AUTO_IMPORT', False)


def auto_import(init_file_path: str):
    # automatically import any Python files in the ofasys directory
    init_root = os.path.dirname(os.path.abspath(init_file_path))
    paths = init_root.split(os.path.sep)
    try:
        module_name = '.'.join(paths[paths.index('ofasys') :])
    except ValueError:
        return

    for filename in sorted(os.listdir(init_root)):
        if os.path.isdir(os.path.join(init_root, filename)) and os.path.exists(
            os.path.join(init_root, filename, '__init__.py')
        ):
            filename = filename
        elif (
            filename.endswith(".py")
            and not filename.startswith("_")
            and not filename.startswith(".")
            and filename != os.path.basename(init_file_path)
        ):
            filename = filename[: filename.find(".py")]
        else:
            continue
        module_file_name = module_name + '.' + filename
        if OFA_LOGGING_AUTO_IMPORT:
            print(f'auto import {module_file_name}')
        importlib.import_module(module_file_name)
