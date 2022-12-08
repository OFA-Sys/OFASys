# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from abc import ABC, abstractmethod
from typing import Any


class BaseReader(ABC):
    """
    The base reader defines basic reader's interface.
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def seek(self, offset: int = 0) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def is_eof(self) -> bool:
        raise NotImplementedError

    def __del__(self):
        self.close()
