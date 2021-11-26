import numpy

from dcflowsim.environment import flow
from dcflowsim.flow_generation import flow_factory
from . import arrival_time_generator_factories, abstract_flow_generator, \
    connection_pair_generator_factories, volume_generator_factories
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import RandomWithConnectionProbabilityFlowGeneratorInterface
from dcflowsim.utils.factory_decorator import factory_decorator
from .flow_generator_factories import FLOW_GENERATOR_FACTORIES


class NodeRemainingVolume(object):
    def __init__(self, node, remaining_volume):
        self.node = node
        self.remaining_volume = remaining_volume


class AbstractFlowGenerator(object):
    def __init__(self):
        self._num_generated_flows = 0

    def reset_num_generated_flows(self):
        """
        Reset the number of generated flows to zero
        Returns:

        """
        self._num_generated_flows = 0


class RandomWithConnectionProbabilityFlowGenerator(AbstractFlowGenerator):
    def __init__(self, connection_pair_generator,
                 volume_generator, arrival_time_generator, flowid_generator, max_num_flows,
                 seed=None, rng=None, flow_type=flow.Flow.__name__):
        """

        Args:
            seed: seed for RNG
            connection_pair_generator: generator for connection pairs
            volume_generator: Generator for flow volumes
            arrival_time_generator: Generator for flow arrival times
            flowid_generator: Generator for global flow IDs
            max_num_flows: Maximum number of flows to generate
            seed: seed of to be created random number generator (can be None)
            rng: random number generator (can be None)
            flow_type: The name of the type of Flow to generate, defaults to Flow, can be SuperFlow
        """
        super(RandomWithConnectionProbabilityFlowGenerator, self).__init__()
        assert (seed is not None or rng is not None), 'both rng and seed cannot be None!'

        self.__seed = seed
        self.rng = numpy.random.RandomState(seed) if rng is None else rng
        self.__connection_pair_generator = connection_pair_generator
        self.__volume_generator = volume_generator
        self.__arrival_time_generator = arrival_time_generator
        self.__flow_id_generator = flowid_generator
        self.__max_num_flows = max_num_flows
        self.__num_generated_flows = 0
        self.__flow_type = flow_type

    def __iter__(self):
        return self

    def __next__(self):
        if self._num_generated_flows >= self.__max_num_flows:
            raise StopIteration
        self._num_generated_flows += 1

        src, dst = self.__connection_pair_generator.__next__()

        return flow_factory.produce_flow(
            self.__flow_type,
            self.__flow_id_generator.get_new_id(),
            self.__arrival_time_generator.__next__(),
            src,
            dst,
            self.__volume_generator.__next__()
        )


@factory_decorator(factory_dict=FLOW_GENERATOR_FACTORIES)
class RandomWithConnectionProbabilityFlowGeneratorFactory(object):
    @classmethod
    def produce(cls, configuration, environment):
        # Create random state
        rng = numpy.random.RandomState(configuration.seed)
        return RandomWithConnectionProbabilityFlowGenerator(
            connection_pair_generator=connection_pair_generator_factories.produce_connection_generator(
                congen_configuration=configuration.connection_generation_configuration,
                environment=environment,
                rng=rng
            ),
            volume_generator=volume_generator_factories.produce(
                volgen_configuration=configuration.volume_generation_configuration,
                rng=rng
            ),
            arrival_time_generator=arrival_time_generator_factories.produce(
                atgen_configuration=configuration.arrival_time_generation_configuration,
                rng=rng
            ),
            flowid_generator=flow.FlowIdGenerator(),
            max_num_flows=configuration.max_num_flows,
            rng=rng,
            flow_type=configuration.flow_type
        )


@peewee_interface_factory_decorator(interface=RandomWithConnectionProbabilityFlowGeneratorInterface)
class RandomWithConnectionProbabilityFlowGeneratorConfiguration(
    abstract_flow_generator.AbstractFlowGeneratorConfiguration):
    def __init__(self, seed, max_num_flows, volume_generation_configuration, arrival_time_generation_configuration,
                 connection_generation_configuration, flow_type=flow.Flow.__name__):
        super(RandomWithConnectionProbabilityFlowGeneratorConfiguration, self).__init__(
            RandomWithConnectionProbabilityFlowGeneratorFactory.__name__
        )
        self.seed = seed
        self.max_num_flows = max_num_flows
        self.connection_generation_configuration = connection_generation_configuration

        self.volume_generation_configuration = volume_generation_configuration
        self.arrival_time_generation_configuration = arrival_time_generation_configuration
        self.flow_type = flow_type

    @property
    def params(self):
        return {
            'seed': self.seed,
            'max_num_flows': self.max_num_flows,
            'connection_generation_configuration': [self.connection_generation_configuration.__class__.__name__,
                                                    self.connection_generation_configuration.params],
            'volume_generation_configuration': [self.volume_generation_configuration.__class__.__name__,
                                                self.volume_generation_configuration.params],
            'arrival_time_generation_configuration': [self.arrival_time_generation_configuration.__class__.__name__,
                                                      self.arrival_time_generation_configuration.params],
            'flow_type': self.flow_type
        }
