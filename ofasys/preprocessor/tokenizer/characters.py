SPACE = chr(32)
SPACE_ESCAPE = chr(9601)


# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


class Characters(object):
    def __init__(self, *unused):
        pass

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def encode(x: str) -> str:
        escaped = x.replace(SPACE, SPACE_ESCAPE)
        return SPACE.join(list(escaped))

    @staticmethod
    def decode(x: str) -> str:
        return x.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)

    def is_beginning_of_word(self, x: str) -> bool:
        return True
