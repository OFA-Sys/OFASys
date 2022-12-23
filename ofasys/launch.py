# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import base64
import codecs
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

import yaml

# Set stdout & stderr encoding to UTF-8
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


WARNINGS_TO_IGNORE = (
    "ignore::UserWarning:torchvision.transforms._functional_video",
    "ignore::UserWarning:torchvision.transforms._transforms_video",
    "ignore:Argument:UserWarning:torchvision.transforms.transforms",
    "ignore:torch.meshgrid:UserWarning:torch.functional",
    "ignore::FutureWarning",
)


def merge_dict(target: Dict, source: Dict):
    if "_name" in target and "_name" in source and target["_name"] != source["_name"]:
        target.clear()
        target.update(source)
        return
    for key, val in source.items():
        if key in target and isinstance(val, dict) and isinstance(target[key], dict):
            merge_dict(target[key], source[key])
        elif key in target and isinstance(target[key], str) and isinstance(val, str) and "${" + key + "}" in val:
            start_pos = val.index("${")
            end_pos = val.index("}", start_pos)
            target[key] = val[:start_pos] + target[key] + val[end_pos + 1 :]
        else:
            target[key] = source[key]


def load_yaml(yaml_path: str) -> Dict:
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path) as f:
        conf = yaml.safe_load(f)
    if "_include" not in conf:
        return conf

    includes = conf["_include"]
    del conf["_include"]
    if isinstance(includes, str):
        includes = [includes]

    result = {}
    for include_cmd in includes:
        include_cmd = include_cmd.split()
        include_path, include_keys = include_cmd[0], include_cmd[1:]

        if not os.path.isabs(include_path):
            include_path = os.path.join(yaml_path, "..", include_path)
        cur_yaml = load_yaml(include_path)

        if len(include_keys) == 0:
            include_yaml = cur_yaml
        else:
            include_yaml = {}
            for include_key in include_keys:
                include_yaml[include_key] = cur_yaml[include_key]
        merge_dict(result, include_yaml)
    merge_dict(result, conf)
    return result


def get_attr(top_conf: Dict, key: str):
    parts = key.split(".")
    cur = top_conf
    for part in parts:
        if part in cur:
            cur = cur[part]
        else:
            raise ValueError(f"Can not find variable {key} in config file.")
    return cur


def variable_substitution(conf: Dict, top_conf=None):
    if top_conf is None:
        top_conf = conf
    result = copy.deepcopy(conf)
    for key, val in conf.items():
        if isinstance(val, dict):
            result[key] = variable_substitution(val, top_conf)
        elif isinstance(val, list):
            if isinstance(val[0], str):
                result[key] = " ||| ".join(val)
            else:
                result[key] = [variable_substitution(v, top_conf) for v in val]
        elif not isinstance(val, str):
            result[key] = val
        else:
            while "${" in val:
                start_pos = val.index("${")
                end_pos = val.index("}", start_pos)
                old_variable = val[start_pos + 2 : end_pos]
                new_variable = get_attr(top_conf, old_variable)
                val = val[:start_pos] + str(new_variable) + val[end_pos + 1 :]
            result[key] = val
    return result


def override_unknown(conf: Dict, unknown_args: List[str]):
    for unknown in unknown_args:
        assert unknown.startswith("--")
        pos = unknown.index("=")
        key, val = unknown[2:pos], unknown[pos + 1 :]
        prefix_key, last_key = key.rsplit(".", 1)
        subconf = get_attr(conf, prefix_key)
        if subconf.get(last_key, None) is not None:
            ori_val = subconf[last_key]
            type_ = type(subconf[last_key])
            if isinstance(ori_val, (int, float, str)):
                # keep the type if its simple enough
                try:
                    subconf[last_key] = type_(val)
                except ValueError as e:
                    print(e)
                    raise ValueError(f"--{key} is supposed to be of type {type_}, but provided {val}") from e
            elif isinstance(ori_val, bool):
                # bool is kind of complicated
                val = val.lower()
                if val in {"true", "t", "yes", "y", "1"}:
                    subconf[last_key] = True
                elif val in {"false", "f", "no", "n", "0"}:
                    subconf[last_key] = False
                else:
                    raise ValueError(f"--{key} is supposed to be of type {type_}, but provided {val}")
            else:
                # this is a complex type, don't know what to do
                raise ValueError(f"--{key} is supposed to be of type {type_}, but provided {val}")

        else:
            # Don't know what to do with these. These are from fairseq or ofasys.
            subconf[last_key] = val


def trim_eval(conf: Dict):
    if conf["env"]["cli"] == "evaluate.py":
        for key in list(conf.keys()):
            if key not in ["env", "task", "model", "common", "checkpoint"]:
                del conf[key]


def check_no_question_mask(conf: Dict):
    for key, val in conf.items():
        if isinstance(val, str) and val == "???":
            raise ValueError(f"{key} is ??? in config, please fill it.")
        elif isinstance(val, dict):
            check_no_question_mask(val)


def extend_star_to_dict(conf):
    assert isinstance(conf, dict)
    if "*" in conf.keys():
        star = conf["*"]
        del conf["*"]
        for key, val in conf.items():
            merge_dict(val, star)
    for key, val in conf.items():
        if isinstance(val, dict):
            extend_star_to_dict(val)


def args_rule(key: str):
    if key.startswith("task") or key.startswith("model"):
        return "ofasys." + key
    else:
        return key.split(".")[-1].replace("_", "-")


def to_args(conf: Dict, prefix: List = None):
    result = []
    if prefix is None:
        prefix = []
    if "_name" in conf:
        if len(prefix) <= 2:
            result.append("--{}='{}'".format(args_rule(".".join(prefix)), conf["_name"]))
        else:
            # TODO: dynamic active?
            result.append("--{}.{}.is_active=True".format(args_rule(".".join(prefix)), conf["_name"]))
        prefix.append(conf["_name"])
    if len(prefix) == 1 and prefix[0] == "task":
        result.append("--{}={}".format(args_rule(".".join(prefix)), ",".join(conf.keys())))
    for key, val in conf.items():
        if key.startswith("_"):
            continue
        prefix.append(key)
        if isinstance(val, dict):
            result.extend(to_args(val, prefix))
        elif isinstance(val, list):
            for i, v in enumerate(val):
                prefix.append(str(i))
                result.extend(to_args(v, prefix))
                prefix.pop()
        elif isinstance(val, bool) and prefix[0] not in ["task", "model"]:
            if val:
                result.append("--{}".format(args_rule(".".join(prefix))))
            else:
                print("ignore fairseq store_false arguments --{}".format(args_rule(".".join(prefix))))
        elif isinstance(val, (int, float, bool)):
            result.append("--{}={}".format(args_rule(".".join(prefix)), val))
        elif isinstance(val, str):
            result.append("--{}='{}'".format(args_rule(".".join(prefix)), val))
        else:
            raise NotImplementedError
        prefix.pop()
    if "_name" in conf:
        prefix.pop()
    return result


def print_conf(conf):
    def _conf_to_text(conf, level=0):
        text = ""
        prefix = "  " * level
        for key in sorted(conf.keys()):
            value = conf[key]
            if isinstance(value, dict):
                text += prefix + f"{key}:\n"
                text += _conf_to_text(value, level + 1)
            else:
                text += prefix + f"{key}: {value}\n"
        return text

    print_box(_conf_to_text(conf), "Launch Configuration")


def print_box(text, title=None, width=80):
    lines = []
    if title:
        title = f" {title.strip()} "
        lines.append(f"╔{title:═^{width-1}}")
    else:
        lines.append("╔" + "═" * (width - 1))
    for line in text.strip().split("\n"):
        lines.append("║" + line)
    lines.append("╚" + "═" * (width - 1))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="N", type=str, nargs="+", help="yaml configs to load")
    parser_args, unknown_args = parser.parse_known_args()

    conf = {}
    for config_path in parser_args.config:
        merge_dict(conf, load_yaml(config_path))

    conf = variable_substitution(conf)
    check_no_question_mask(conf)
    override_unknown(conf, unknown_args)
    trim_eval(conf)
    extend_star_to_dict(conf)
    print_conf(conf)

    env = conf["env"]
    del conf["env"]
    args = ['--ofasys_complete_config=' + base64.b64encode(json.dumps(conf).encode()).decode()]
    if 'extra_models' in conf['model']:
        del conf['model']['extra_models']
    args += to_args(conf)

    bash_cmd = ""
    if env["runner"] == "local":
        cli_path = os.path.abspath(os.path.join(__file__, "../cli/" + env["cli"]))
        config_path = os.path.abspath(os.path.join(parser_args.config[0], ".."))
        bash_cmd += f"set -euo pipefail\ncd {config_path}\n"
        # add the path of codebase into PYTHONPATH
        ofasys_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        bash_cmd += f"export PYTHONPATH={ofasys_dir_name}\n"
        # configure distrubuted environment
        assert env["nnodes"] >= 1
        if env["nnodes"] > 1:
            bash_cmd += f"export MASTER_PORT={env['port']}\n"
        if env["nproc_per_node"] > 1 and "OMP_NUM_THREADS" not in os.environ:
            nthread = max(1, int(len(os.sched_getaffinity(0)) / env["nproc_per_node"]))
            bash_cmd += f"export OMP_NUM_THREADS={nthread}\n"
        if env["cuda_visible_devices"] is not None:
            bash_cmd += f"export CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}\n"
        if env.get("ofa_cache_home", None) is not None:
            bash_cmd += f'export OFA_CACHE_HOME={env["ofa_cache_home"]}\n'
        bash_cmd += f'export PYTHONWARNINGS={",".join(WARNINGS_TO_IGNORE)}\n'
        if env["nnodes"] == 1 and env["nproc_per_node"] == 1:
            bash_cmd += f"python3 {cli_path}"
        elif env["nnodes"] == 1:
            bash_cmd += (
                "python3 -m torch.distributed.launch --nproc_per_node="
                f"{env['nproc_per_node']} --master_port={env['port']} {cli_path}"
            )
        else:
            bash_cmd += (
                f"python3 -m torch.distributed.launch --node_rank={env['rank']} "
                f"--nnodes={env['nnodes']} --nproc_per_node={env['nproc_per_node']} "
                f"--master_addr={env['master_addr']} --master_port={env['port']} "
                f"{cli_path}"
            )
        for arg in args:
            bash_cmd += f" \\\n\t{arg}"
    elif env["runner"] == "dlc":
        cli_path = os.path.abspath(os.path.join(__file__, "../cli/" + env["cli"]))
        bash_cmd += f"set -euo pipefail\n"
        # add the path of codebase into PYTHONPATH
        ofasys_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        bash_cmd += f"export PYTHONPATH={ofasys_dir_name}\n"
        # filter future warning
        bash_cmd += f'export PYTHONWARNINGS={",".join(WARNINGS_TO_IGNORE)}\n'
        # the master port, master addr and node rank will be overrided by environment vars of DLC workers
        master_port = os.environ["MASTER_PORT"] if "MASTER_PORT" in os.environ else "6000"
        master_addr = os.environ["MASTER_ADDR"] if "MASTER_ADDR" in os.environ else "localhost"
        node_rank = os.environ["RANK"] if "RANK" in os.environ else "0"
        nnodes = os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else "1"
        status, nproc_per_node = subprocess.getstatusoutput(
            'python3 -c "import torch; print(torch.cuda.device_count())"'
        )
        assert status == 0 and int(nproc_per_node) > 0, "Cannot find any cuda device in DLC"
        bash_cmd += (
            f"python3 -m torch.distributed.launch --node_rank={node_rank} "
            f"--nnodes={nnodes} --nproc_per_node={nproc_per_node} "
            f"--master_addr={master_addr} --master_port={master_port} "
            f"{cli_path}"
        )
        for arg in args:
            bash_cmd += f" \\\n\t{arg}"
    else:
        raise ValueError(f"unsupported env.runner, get {env['runner']}, expect 'dlc', or 'local'")

    print_box(bash_cmd, "Launch Commands")

    pipe = subprocess.Popen(bash_cmd, shell=True, executable="/bin/bash")
    pipe.wait()
