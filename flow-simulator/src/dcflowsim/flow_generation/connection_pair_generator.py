import abc
import numpy as np

from dcflowsim import configuration
from dcflowsim.network.switch_types import SwitchTypes
from dcflowsim.network import network_elements
from dcflowsim.utils.factory_decorator import factory_decorator
from .connection_pair_generator_factories import CONNECTION_PAIR_GENERATOR_FACTORY


class AbstractConnectionPairGenerator(abc.ABC):
    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


class TorConnectionPairUniformGenerator(AbstractConnectionPairGenerator):
    """ Generates tor connection pairs from uniformly """

    def __init__(self, topo_rack_nodes, rng=None, seed=None):
        """
        Construct Generator.

        Args:
            seed(int): Seed for the rng
        """
        assert (rng is not None or seed is not None)

        self.rng = np.random.RandomState(seed) if rng is None else rng
        self._topo_rack_node = topo_rack_nodes

    def __next__(self):
        """

        Returns:
            src(str), dst(str)
        """
        nodes = range(len(self._topo_rack_node))
        src = self.rng.choice(nodes)
        dst = self.rng.choice(nodes)

        while src == dst:
            dst = self.rng.choice(nodes)

        return self._topo_rack_node[src], self._topo_rack_node[dst]


class HostLevelTorConnectionPairUniformGenerator(TorConnectionPairUniformGenerator):
    """
    Generates Host Level Tor connection pairs uniformly. Sets actual flow endpoints on hosts in the racks
    """

    def __init__(self, hosts_per_rack, rng=None, seed=None):
        """

        Args:
            hosts_per_rack: dict, hosts per rack
            rng: rng
            seed: int, seed for the random number generator
        """
        super(HostLevelTorConnectionPairUniformGenerator, self).__init__(
            topo_rack_nodes=list(hosts_per_rack.keys()),
            seed=seed,
            rng=rng,
        )

        self._num_racks = len(hosts_per_rack)
        self._hosts_per_rack = hosts_per_rack

    @property
    def num_racks(self):
        return self._num_racks

    def __next__(self):
        """

        Returns: str, source and destination host pairs

        """
        rack_pair = super(HostLevelTorConnectionPairUniformGenerator, self).__next__()

        if rack_pair is None:
            return None

        src_rack, dst_rack = rack_pair
        src_host = self.rng.choice(self._hosts_per_rack[src_rack])
        dst_host = self.rng.choice(self._hosts_per_rack[dst_rack])

        return src_host, dst_host


class HostLevelTorConnectionPairUniformGeneratorConfiguration(configuration.AbstractConfiguration):
    """
    Configures the host level uniform Tor pair generator
    """

    def __init__(self, tor_switch_identifier=None, seed=None):
        """

        Args:
            seed: int, seed for the rng
        """
        super(HostLevelTorConnectionPairUniformGeneratorConfiguration, self).__init__(
            HostLevelTorConnectionPairUniformGeneratorFactory.__name__)
        self.seed = seed
        self.tor_switch_identifier = tor_switch_identifier if tor_switch_identifier is not None else SwitchTypes.edge

    @property
    def params(self):
        return {
            'seed': self.seed,
            'tor_switch_identifier': self.tor_switch_identifier
        }


@factory_decorator(factory_dict=CONNECTION_PAIR_GENERATOR_FACTORY)
class HostLevelTorConnectionPairUniformGeneratorFactory(object):
    """
    Host level Tor Connection Pair Generator Factory for uniform generation
    """

    def produce(self, config, environment, rng=None):
        """

        Args:
            config: HostLevelTorConnectionPairUniformGeneratorConfiguration
            environment: simulation environment
            rng: rng

        Returns: HostLevelTorConnectionPairUniformGenerator

        """
        racks = environment.topology.get_nodes_by_type(config.tor_switch_identifier)
        assert len(racks) > 1
        assert (rng is not None or config.seed is not None), "Either a seed or a rng must be provided"

        hosts_per_rack = dict()
        for rack in racks:
            hosts_per_rack[rack.node_id] = []
            for node_id in rack.physical_neighbors:
                node = environment.topology.get_node(node_id)
                if isinstance(node, network_elements.Host):
                    hosts_per_rack[rack.node_id].append(node)

        my_gen = HostLevelTorConnectionPairUniformGenerator(
            hosts_per_rack=hosts_per_rack,
            rng=rng if rng is not None else config.seed
        )
        return my_gen
