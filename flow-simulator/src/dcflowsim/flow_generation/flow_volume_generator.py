from abc import ABC

import numpy

from dcflowsim.utils.number_conversion import convert_to_decimal
from ..utils.factory_decorator import factory_decorator
from .volume_generator_factories import VOLUME_GENERATOR_FACTORIES


class AbstractFlowVolumeGeneratorConfiguration(ABC):
    def __init__(self, factory):
        self.factory = factory

    @property
    def params(self):
        """
        Returns a dict containing the parameters
        Returns:

        """
        raise NotImplementedError


class CdfBasedFlowVolumeGeneratorConfiguration(AbstractFlowVolumeGeneratorConfiguration):
    def __init__(self, values, cdf_values, seed):
        super(CdfBasedFlowVolumeGeneratorConfiguration, self).__init__(
            CdfBasedFlowVolumeGeneratorFactory.__name__)
        self.values = values
        self.cdf_values = cdf_values
        self.seed = seed

    @property
    def params(self):
        return {
            'values': self.values,
            'cdf_values': self.cdf_values,
            'seed': self.seed
        }


@factory_decorator(factory_dict=VOLUME_GENERATOR_FACTORIES)
class CdfBasedFlowVolumeGeneratorFactory(object):
    @classmethod
    def produce(cls, configuration, rng=None):
        # If seed is set in configuration, use it
        if configuration.seed is not None:
            return CdfBasedFlowVolumeGenerator(
                configuration.values,
                configuration.cdf_values,
                configuration.seed
            )
        else:
            return CdfBasedFlowVolumeGenerator(
                values=configuration.values,
                cdf_values=configuration.cdf_values,
                rng=rng
            )


class CdfBasedFlowVolumeGenerator(object):
    """
    Generates uniformly distributed flow volumes
    """

    def __init__(self, values, cdf_values, seed=None, rng=None):
        """

        Args:
            values: array containing the potential sample values
            cdf_values: array/list containing the accumulated probabilities of the sample values (CDF)
            seed: Seed for RNG
            rng: numpy RandomState. Is preferred over the seed
        """
        assert (rng is not None or seed is not None), "both rng and seed cannot be None!"

        self.__values = values
        self.__cdf_values = numpy.array(cdf_values)
        self.__seed = seed
        self.rng = numpy.random.RandomState(seed) if rng is None else rng

    def __iter__(self):
        return self

    def __next__(self):
        r = self.rng.random_sample()
        # where difference between random number and cdf_value (cum probability) > 0
        candidate_indices = (self.__cdf_values - r) > 0
        potential_cdf_values = self.__cdf_values[candidate_indices]
        smallest_cdf_value = self.__cdf_values == min(  # smallest value
            potential_cdf_values
        )
        index_smallest_cdf_value = numpy.argwhere(  # index of
            smallest_cdf_value
        )[0]
        return convert_to_decimal(int(
            self.__values[int(index_smallest_cdf_value)]
        ))
