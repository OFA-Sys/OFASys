# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from functools import partial
from typing import Callable, Dict, List, Optional

from ofasys.utils.logging_utils import master_logging

from .configs import BaseDataclass
from .parser import _getattr, _hasattr, _parser_add_dataclass, _setattr
from .singleton import Singleton

logger = logging.getLogger(__name__)


def register_config(group: str, name: str, dataclass: Optional[type] = None) -> Callable:
    def _register(cls):
        ConfigStore().store(group, name, cls, dataclass)
        return cls

    return _register


@dataclass
class ConfigNode:
    target: object
    config: Optional["dataclass"] = None
    is_active: bool = False


class ConfigStore(object, metaclass=Singleton):
    """Store OFASys related configurations

    All configuration nodes are gathered in the module import phase;
    Then ofasys related args are add to the main parser;
    Then user specified configurations are imported from arguments;
    The related config nodes are overrided with given arguments.
    """

    repo: Dict[str, ConfigNode]

    def __init__(self) -> None:
        self.repo = dict()

    def _get_group_name(self, path: str):
        """
        Top-level groups are:
        ofasys.model: only has unify
        ofasys.task
        """
        pos = path.rindex(".")
        return path[:pos], path[pos + 1 :]

    def store(self, group: str, name: str, obj: type, dc: Optional[type] = None) -> None:
        """Record the existence of the config named <group>.<name>

        If dc is given, a default configuration is initialized
        """
        assert group and name
        assert dc is None or (isinstance(dc, type) and is_dataclass(dc))
        path = f"{group}.{name}"
        assert path not in self.repo
        if dc is not None:
            self.repo[path] = ConfigNode(obj, dc())
        else:
            self.repo[path] = ConfigNode(obj)

    def get_dict(self, group_key: str) -> Dict[str, ConfigNode]:
        """Get a dict of ConfigNode with the name as key for a group"""
        candidate = {}
        for path, node in self.repo.items():
            group, name = self._get_group_name(path)
            if group_key == group and node.is_active:
                candidate[name] = self.repo[path]
        return candidate

    def get(self, group_key: str, name: Optional[str] = None) -> ConfigNode:
        """Get the group"""
        if name is None:
            candidate = []
            for path, node in self.repo.items():
                group, name = self._get_group_name(path)
                if group_key == group and node.is_active:
                    candidate.append(path)
            return [self.repo[path] for path in candidate]
        else:
            return self.repo[f"{group_key}.{name}"]

    def contain(self, group_key: str, name: str) -> bool:
        """Check if the group exists"""
        path = f"{group_key}.{name}"
        return path in self.repo

    def build(self, group_key, name=None) -> Callable:
        """Get the group related classes constructor with config passed"""
        nodes = self.get(group_key, name)
        if name is not None:
            nodes = [nodes]
        results = []
        for node in nodes:
            if node.config:
                results.append(partial(node.target, cfg=node.config))
            else:
                results.append(node.target)
        if name is not None:
            return results[0]
        return results

    def _active(self, parser):
        """Add group-level arguments to the parser and parse the group

        This takes place before dist init
        """
        candidate = defaultdict(list)

        # get available groups and their possible choices
        for path in self.repo.keys():
            group, name = self._get_group_name(path)
            assert group != 'ofasys', f'Do not allow single-choice config node for now, got {group}'
            candidate[group].append(name)

        # parse groups
        pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        for group, choices in candidate.items():
            pre_parser.add_argument("--" + group, type=str, default=None, help="candidate: " + ",".join(choices))
            parser.add_argument("--" + group, type=str, default=None, help="candidate: " + ",".join(choices))
        parsed_args, _ = pre_parser.parse_known_args()

        # make the group active
        for group, names in vars(parsed_args).items():
            if names is not None:
                for name in names.split(","):
                    self.repo[f"{group}.{name}"].is_active = True
                    logger.info(f"ConfigStore activates {group}.{name}")

    @master_logging()
    def add_args(self, parser: argparse.ArgumentParser):
        """Add OFASys related arguments to the main fairseq parser

        First, the group level arguments --ofasys.task= is added and the group parsed
        and then, the args related to the group are added
        """

        # parse active groups
        self._active(parser)

        # add active group args
        for prefix, node in self.repo.items():
            if node.config is not None and node.is_active:
                group = parser.add_argument_group(f"{prefix} configuration")
                _parser_add_dataclass(group, node.config, None, prefix)

    @master_logging()
    def import_args(self, args: argparse.Namespace):
        """Import OFASys related arguments from parsed args

        The following arguments are considered ofasys related:
        1. indicated by the prefix ofasys.
        2. known by the config store, i.e., the related module is imported and registered

        For repeated arguments, last occurence wins
        The given argument overrides the default values in the config node
        """
        vars_args = list(vars(args).items())

        for key, val in vars_args:

            # ofasys related arguments have two forms
            # 1. --ofasys.task=mytask
            # 2. --ofasys.task.mytask.xx.xxx=xxxx
            # the former specifies which config instance should be used for the group
            # the latter specifies the config of the instance
            # is_set is True for the former
            is_set = False

            for path, node in self.repo.items():
                # path is of the form ofasys.task.my_task
                # of which, ofasys.task is the group and my_task is the name
                group, name = self._get_group_name(path)
                if group == key:
                    # this is the form like --ofasys.task==mytask
                    is_set = True
                    break

                assert key != path
                if key.startswith(path):
                    # this is the form like --ofasys.task.mytask.xx.xxx=xxxx
                    # suffix key is xx.xxx, value is xxxx
                    # as the configs are hierarchical, needs recursive find and set
                    suffix_key = key[len(path) :].strip(".")
                    if _hasattr(node.config, suffix_key):
                        _setattr(node.config, suffix_key, val)
                        is_set = True
                        break

            if not is_set and key.startswith("ofasys."):
                logger.info(f"unrecognized argument: {key}")

        self.print()

    def make_dataclass(self, group_key, dc_name, module_name, prefix_names=[]) -> BaseDataclass:
        fields = []

        def cmp(val):
            path = val[0]
            _, name = self._get_group_name(path)
            if name in prefix_names:
                return (prefix_names.index(name), None)
            else:
                return (len(prefix_names), path)

        for path, node in sorted(self.repo.items(), key=cmp):
            group, name = self._get_group_name(path)
            if group == group_key and node.config is not None:
                fields.append(
                    (
                        name,
                        type(node.config),
                        field(default_factory=node.config.__class__),
                    )
                )
        config_cls = make_dataclass(dc_name, fields, bases=(BaseDataclass,))
        config_cls.__module__ = module_name
        return config_cls

    def print(self, width=80, ignore_inactive=True):
        """Print all active configurations in a right open box"""
        if ignore_inactive:
            all_active_nodes = [(path, node) for path, node in self.repo.items() if node.is_active]
        else:
            all_active_nodes = [(path, node) for path, node in self.repo.items() if node.config]
        all_active_nodes.sort(key=lambda x: x[0])

        lines = []
        for i, (path, node) in enumerate(all_active_nodes):
            title = f" {path} {node.target.__name__} {node.config.__class__.__name__} "
            if i == 0:
                lines.append(f"╔{title:═^{width-1}}")
            else:
                lines.append(f"╠{title:═^{width-1}}")
            lines.extend(self._print_dataclasses(node.config, level=0, ignore_inactive=ignore_inactive))
        lines.append("╚" + "═" * (width - 1))

        logger.info("OFASys related configurations:\n" + "\n".join(lines))

    @staticmethod
    def _print_dataclasses(dc: dataclass, level: int = 0, ignore_inactive=True) -> List[str]:

        prefix = "║" + "  " * level

        all_fields = [field for field in dataclasses.fields(dc) if not field.name.startswith("_")]
        all_fields.sort(
            key=lambda field: (
                dataclasses.is_dataclass(getattr(dc, field.name)),
                field.name,
            )
        )

        lines = []
        for field in all_fields:
            key = field.name
            value = getattr(dc, key)
            if dataclasses.is_dataclass(value):
                # metric uses target_field, while many others use is_active
                if not ignore_inactive or (getattr(value, "is_active", True) and getattr(value, "target_field", True)):
                    lines.append(f"{prefix}{key} ({value.__class__.__name__}):")
                    lines.extend(
                        ConfigStore._print_dataclasses(value, level=level + 1, ignore_inactive=ignore_inactive)
                    )
            else:
                lines.append(f"{prefix}{key}: {value}")
        return lines
