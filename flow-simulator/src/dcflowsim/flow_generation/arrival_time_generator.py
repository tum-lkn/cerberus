from abc import ABC
import numpy
from decimal import Decimal
from dcflowsim.utils.number_conversion import convert_to_decimal
from dcflowsim.utils.factory_decorator import factory_decorator
from .arrival_time_generator_factories import ARRIVAL_TIME_GENERATOR_FACTORIES


class AbstractArrivalTimeGeneratorConfiguration(ABC):
    def __init__(self, factory):
        self.factory = factory

    @property
    def params(self):
        """
        Returns a dict containing the parameters
        Returns:

        """
        raise NotImplementedError


class PoissonArrivalTimeGenerator(object):
    """
    Generates uniformly distributed flow volumes
    """

    def __init__(self, mean_inter_arrival_time, seed=None, rng=None):
        """

        Args:
            mean_inter_arrival_time: mean inter arrival time (lambda)
        """
        assert (rng is not None or seed is not None)
        self.__mean_inter_arrival_time = mean_inter_arrival_time
        self.rng = numpy.random.RandomState(seed) if rng is None else rng
        self.__arrival_time = Decimal('0')

    def __iter__(self):
        return self

    def __next__(self):
        self.__arrival_time = \
            self.__arrival_time + \
            convert_to_decimal(self.rng.exponential(self.__mean_inter_arrival_time))
        return self.__arrival_time


class PoissonArrivalTimeGeneratorConfiguration(AbstractArrivalTimeGeneratorConfiguration):
    def __init__(self, mean_inter_arrival_time, seed):
        super(PoissonArrivalTimeGeneratorConfiguration, self).__init__(PoissonArrivalTimeGeneratorFactory.__name__)
        self.mean_inter_arrival_time = mean_inter_arrival_time
        self.seed = seed

    @property
    def params(self):
        return {
            "mean_iat": self.mean_inter_arrival_time,
            "seed": self.seed
        }


@factory_decorator(factory_dict=ARRIVAL_TIME_GENERATOR_FACTORIES)
class PoissonArrivalTimeGeneratorFactory(object):
    @classmethod
    def produce(cls, configuration, rng=None):
        # If seed is set in configuration, use it
        if configuration.seed is not None:
            return PoissonArrivalTimeGenerator(configuration.mean_inter_arrival_time, configuration.seed)
        else:
            return PoissonArrivalTimeGenerator(configuration.mean_inter_arrival_time, rng=rng)
