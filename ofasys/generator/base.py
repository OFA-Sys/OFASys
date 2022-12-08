# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn as nn


def to_numpy(x):
    if x is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


@dataclass
class GeneratorOutput(object):
    """
    Base Class for output of Generator.
    Users can directly get the output of the supported modality through calling the modality name.
    This class will also provide interfaces to dump output data as a file (e.g. png, wav, gif).
    Original output in tensor format and extra information are also provided.
    """

    pass


MultiGeneratorOutput = List[GeneratorOutput]
BatchGeneratorOutput = List[Union[GeneratorOutput, MultiGeneratorOutput]]


class Generator(nn.Module):
    def __init__(self):
        """
        Base Class for Generator.
        As OFASys follows the encoder-decoder architecture, the output is mainly produced by the decoder.
        However, as the decoder accepts both E-slots and D-slots of various slot types,
        different generators are provided in OFASys to complement the differences in generation paradigm.
        """
        super().__init__()

    def forward(self, model, sample, **kwargs):
        """Calls the ``generate()`` function to generate result by default.

        Parameters
        ----------
        model: the model object.
        sample: preprocessed batch input data.
        """
        return self.generate(model, sample, **kwargs)

    @torch.no_grad()
    @abstractmethod
    def generate(self, model, sample, **kwargs):
        """Generate function. Should be overridden by all subclasses.

        Parameters
        ----------
        model: the model object.
        sample: preprocessed batch input data.


        """
        raise NotImplementedError
