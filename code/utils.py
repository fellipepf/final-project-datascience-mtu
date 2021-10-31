from enum import Enum, IntEnum


class Target(IntEnum):
    PATIENT = 1
    CONTROL = 0

    def __str__(self):
        return self.name