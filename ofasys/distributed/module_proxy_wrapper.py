# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from torch import nn


class ModuleProxyWrapper(nn.Module):
    """
    Wrap a DistributedDataParallel module and forward requests for missing
    attributes to the module wrapped by DDP (the twice-wrapped module).
    Also forward calls to :func:`state_dict` and :func:`load_state_dict`.

    Usage::

        module.xyz = "hello world"
        wrapped_module = DistributedDataParallel(module, **ddp_args)
        wrapped_module = ModuleProxyWrapper(wrapped_module)
        assert wrapped_module.xyz == "hello world"
        assert wrapped_module.state_dict().keys() == module.state_dict().keys()

    Args:
        module (nn.Module): module to wrap
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        assert hasattr(module, "module"), "ModuleProxyWrapper expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # forward to the once-wrapped module
                return getattr(self.module, name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
