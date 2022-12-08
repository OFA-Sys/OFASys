# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import Union

import torch.distributed as dist


def is_logging_master() -> bool:
    """Check if this process is the rank_zero heuristically.

    This is useful before dist initailized. Currently only support torch.distributed.lanuch or
    torch.distribtued.run style.
    """

    # via dist
    if dist.is_initialized():
        return dist.get_rank() == 0

    # via env
    if "RANK" in os.environ:
        # This is torch.distributed.run or torch.distribued.launch with use_env.
        return os.environ["RANK"] == "0"

    # via cmd args
    # This torch.distributed.lanuch, which only pass --local_rank=xx and
    # we have no way reliable to determine the node rank.
    # Use local_rank instead
    m = re.search("--local_rank=(\d+)", " ".join(sys.argv))
    if m:
        # This is torch.distributed.lanuch, which only pass --local_rank=xx.
        # We have no reliable way to determine the node rank.
        local_rank = m.group(1)
        return local_rank == "0"

    # cannot determine rank, logging on all rank is fine
    return True


@contextmanager
def master_logging(level: int = logging.WARNING):
    """Make logging less nosiy by logging only on the logging master but still
    keep the message above level. This is the same level with fairseq.

    Note: when use as decorators, don't forget the final parenthese.
    """
    # We use logging.disable because this is only useful before dist.init (after
    # that, fairseq will have taken care of logging) and most logs are from
    # libraries, which may have set its own logging level.
    cond = is_logging_master()
    prev_level = logging.root.manager.disable
    try:
        if not cond:
            logging.disable(level)
        yield
    finally:
        if not cond:
            logging.disable(prev_level)


def reconfig_logger(
    logger: logging.Logger,
    format: str = "%(asctime)s - %(name)s@%(lineno)d - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%dT%H:%M:%S",
    level: Union[str, int] = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
) -> logging.Logger:
    """Reconfigure the given logger with the OFA-Sys style."""
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    h = logging.StreamHandler(stream)
    handlers = [h]
    fmt = logging.Formatter(format, datefmt, "%")
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
