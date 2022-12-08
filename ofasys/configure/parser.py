# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import argparse
import ast
import re
from dataclasses import _MISSING_TYPE, MISSING, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def _dc_get_default(dc, attribute_name: str) -> Any:
    if hasattr(dc, attribute_name):
        if str(getattr(dc, attribute_name)).startswith("${"):
            return str(getattr(dc, attribute_name))
        elif str(dc.__dataclass_fields__[attribute_name].default).startswith("${"):
            return str(dc.__dataclass_fields__[attribute_name].default)
        elif getattr(dc, attribute_name) != dc.__dataclass_fields__[attribute_name].default:
            return getattr(dc, attribute_name)

    f = dc.__dataclass_fields__[attribute_name]
    if not isinstance(f.default_factory, _MISSING_TYPE):
        return f.default_factory()
    return f.default


def _dc_get_meta(dc, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
    return dc.__dataclass_fields__[attribute_name].metadata.get(meta, default)


def _dc_get_help(dc, attribute_name: str) -> Any:
    return _dc_get_meta(dc, attribute_name, "help")


def _dc_get_alias(dc, attribute_name: str) -> Any:
    return _dc_get_meta(attribute_name, "argparse_alias")


def _interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(r"(typing.|^)Union\[(.*), NoneType\]$", typestring) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type


def _eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def _eval_bool(x):
    if x is None:
        return False
    elif isinstance(x, bool):
        return x
    elif isinstance(x, str):
        x = x.lower()
        assert x in ['true', 'false']
        return x == 'true'
    else:
        raise ValueError


def _rshrink(key: str) -> str:
    assert key.strip('.') == key
    idx = key.rfind('.')
    while idx >= 0:
        yield key[:idx], key[idx + 1 :]
        idx = key.rfind('.', 0, idx)


def _gen_arg_name(field_name: str, alias_map: Dict[str, str], prefix: str = ""):
    args = []
    long_field_name = "--" + (f"{prefix}.{field_name}" if prefix else field_name)
    args.append(long_field_name)

    # if field_name in alias_map:
    #     for _, suffix_key in _rshrink(f"{prefix}.{field_name}"):
    #         if suffix_key not in alias_map:
    #             field_alias = "--" + suffix_key
    #             alias_map[suffix_key] = long_field_name
    #             break
    #     else:
    #         field_alias = None
    # else:
    #     field_alias = "--" + field_name
    #     alias_map[field_name] = long_field_name
    # if field_alias:
    #     args.append(field_alias)

    # if '_' in field_name:
    #     args.extend(_gen_arg_name(field_name.replace('_', '-'), alias_map, prefix))
    return args


def _parser_add_dataclass(
    parser: argparse.ArgumentParser,
    dc: Any,
    alias_map: Dict[str, str],
    prefix: str = "",
) -> None:

    if not is_dataclass(dc):
        raise ValueError(f"dc must be a dataclass, got {dc}.")

    for field_name, field_val in dc.__dataclass_fields__.items():
        field_type = field_val.type
        if field_name.startswith('_'):
            continue
        elif is_dataclass(field_type):
            nested_prefix = f'{prefix}.{field_name}' if prefix else field_name
            _parser_add_dataclass(parser, getattr(dc, field_name), alias_map, nested_prefix)
            continue

        inter_type = _interpret_dc_type(field_type)
        field_default = _dc_get_default(dc, field_name)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        kwargs = {}
        field_help = _dc_get_help(dc, field_name)
        kwargs["help"] = field_help

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (isinstance(inter_type, type) and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))) or (
                "List" in str(inter_type) or "Tuple" in str(inter_type)
            ):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: _eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: _eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: _eval_str_list(x, str)
                else:
                    raise NotImplementedError("parsing of type " + str(inter_type) + " is not implemented")
                if field_default is not MISSING:
                    kwargs["default"] = ",".join(map(str, field_default)) if field_default is not None else None
            elif (isinstance(inter_type, type) and issubclass(inter_type, Enum)) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["type"] = _eval_bool
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        # del kwargs["default"]
        args = _gen_arg_name(field_name, alias_map, prefix)
        parser.add_argument(*args, **kwargs)


def _getattr(obj, attr, default=None):
    try:
        left, right = attr.split('.', 1)
    except Exception:
        return getattr(obj, attr, default)
    return _getattr(getattr(obj, left), right, default)


def _setattr(obj, attr, val):
    try:
        left, right = attr.split('.', 1)
    except Exception:
        return setattr(obj, attr, val)
    return _setattr(getattr(obj, left), right, val)


def _hasattr(obj, attr):
    try:
        left, right = attr.split('.', 1)
    except Exception:
        return hasattr(obj, attr)
    return _hasattr(getattr(obj, left), right)
