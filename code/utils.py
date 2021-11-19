from enum import Enum, IntEnum


class Target(IntEnum):
    PATIENT = 1
    CONTROL = 0

    def __str__(self):
        return self.name

class Feature:
    MEAN = "mean"
    SD = "sd"
    PROP_ZEROS = "prop_zero"
    KURTOSIS = "kurtosis"
    SKEW = "skew"

    def __str__(self):
        return self.name

if __name__ == '__main__':
    print(Feature.KURTOSIS)