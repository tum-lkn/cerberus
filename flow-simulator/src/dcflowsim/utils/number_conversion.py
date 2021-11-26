"""
Implements conversion from python floats and ints to well defined python Decimal objects
"""

__version__ = 0.0
__author__ = "Sebastian Schmidt, ga96vog@mytum.de"

from decimal import *
from dcflowsim import constants


def convert_to_decimal(to_convert, decimal_places=constants.FLOAT_DECIMALS):
    """
    Converts a floating point number into a Decimal object by rounding to a number of
    decimal digits specified in constants. A Decimal object's precision is not affected.

    Args:
        to_convert: The float to be converted
        decimal_places: Nr decimal places for rounding, default: constants.FLOAT_DECIMALS

    Returns:
        A Decimal object with a rounded representation
    """

    if isinstance(to_convert, Decimal):
        return to_convert
    else:
        return Decimal("{number:.{decimal_places}f}".
                       format(number=to_convert, decimal_places=decimal_places))


def round_decimal_up(decimal_to_round, decimal_places=constants.FLOAT_DECIMALS):
    """ Rounds a Decimal number to a fixed number of decimal places with an always up
    rounding convention. Returns a modified object.

    Args:
        decimal_to_round: Decimal object to round
        decimal_places: number of decimal places to round to
    """
    return Decimal(decimal_to_round).quantize(Decimal("1." + ("0" * decimal_places)),
                                              rounding=ROUND_UP)


def round_decimal_down(decimal_to_round, decimal_places=constants.FLOAT_DECIMALS):
    """ Rounds a Decimal number to a fixed number of decimal places with an always down
    rounding convention. Returns a modified object.

    Args:
        decimal_to_round: Decimal object to round
        decimal_places: number of decimal places to round to
    """
    return Decimal(decimal_to_round).quantize(Decimal("1." + ("0" * decimal_places)),
                                              rounding=ROUND_DOWN)


def round_decimal(decimal_to_round, decimal_places=constants.FLOAT_DECIMALS):
    """ Rounds a decimal according to context rounding convention. Default
    setting from decimal is ROUND_HALF_EVEN."""
    return Decimal(decimal_to_round).quantize(Decimal("1." + ("0" * decimal_places)))