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


def get_name_from_value(value):
    try:
        value = int(float(value))

        if value in [0, 1]:
            return Target(value).name
        else:
            return value
    except ValueError:
        return value


if __name__ == '__main__':
    print(Feature.KURTOSIS)

    print(Target.PATIENT.value)
    print(Target(1).name)
    print(Target(0).name)
    print(get_name_from_value("kk"))
    print(get_name_from_value("1.0"))
    print(get_name_from_value("1"))
    print(get_name_from_value("3"))