import networkx as nx
from dcflowsim.network.topology import AbstractTopologyFactory, AbstractTopologyConfiguration, BaseDataCenterTopology
from dcflowsim.network.networkx_wrapper import NetworkXTopologyWrapper
from dcflowsim.network.network_elements import Host, LinkIdentifiers
from dcflowsim.network.network_elements_factory import produce_switch
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import TopologyInterface
from dcflowsim.utils.factory_decorator import factory_decorator
from .topology_factories import TOPOLOGY_FACTORIES


@factory_decorator(factory_dict=TOPOLOGY_FACTORIES)
class ExpanderNetTopologyFactory(AbstractTopologyFactory):
    """
    Factory class for ExpanderNet Topologies.

    Each Rack contains the same number of hosts. The topology has the structure of a randomly generated expander
    graph.
    """

    @classmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        """
        Factory function. Creates all hosts, electrical switches and links.
        Args:
            topology_configuration: ExpanderNetTopologyConfiguration, config for the ExpanderNet topology
            decorated_topology: Is ignored for ExpanderNets as this type of topology cannot decorate others.

        Returns: A Expander topology; Really a BaseDatacenterTopology with the expandernet structure.

        It just generates a d-regular, random graph

        """
        assert decorated_topology is None, "Cannot decorate another topology"

        expandernet_topo = BaseDataCenterTopology(network_graph_wrapper=topology_configuration.networkx_wrapper_type())

        # Add the ToR switches, hosts and add electrical links between them
        for rack_id in range(topology_configuration.nr_racks):
            tor_switch = produce_switch(
                config=topology_configuration.tor_switch_config,
                node_id=f"TorSwitch-{rack_id}",
                num_ports=topology_configuration.nr_hosts_per_rack + topology_configuration.nr_links,
                network_identifier="TorSwitch"
            )
            expandernet_topo.add_switch(tor_switch)

            for host_id in range(topology_configuration.nr_hosts_per_rack):
                host = Host(node_id=f"Host-P{rack_id},{host_id}")
                expandernet_topo.add_host(host)

                expandernet_topo.add_link(
                    node_1=tor_switch,
                    node_2=host,
                    capacity=topology_configuration.host_link_capacity
                )

        if topology_configuration.nr_links == 0:
            # No static switch so just return ToRs with Hosts
            return expandernet_topo

        # Generate nr_links-regular graph with nr_racks nodes
        if topology_configuration.nr_links > 2:
            dregular_graph = nx.random_regular_graph(
                topology_configuration.nr_links,
                topology_configuration.nr_racks,
                topology_configuration.seed
            )
        elif topology_configuration.nr_links > 0:
            dregular_graph = nx.grid_graph(dim=(topology_configuration.NR_RACKS,))
        else:
            raise RuntimeError("Should not reach this end.")
        assert nx.is_connected(dregular_graph)
        # Add links for graph to our topology. Match racks and node ids using idx in switches list
        racks = expandernet_topo.get_switches()
        for edge in dregular_graph.edges.keys():
            expandernet_topo.add_link(
                node_1=racks[edge[0]],
                node_2=racks[edge[1]],
                capacity=topology_configuration.tor_switch_config.link_capacity,
                link_identifier=LinkIdentifiers.static
            )
        return expandernet_topo


@peewee_interface_factory_decorator(interface=TopologyInterface)
class ExpanderNetTopologyConfiguration(AbstractTopologyConfiguration):
    """
    Configuration for a ExpanderNet topology. ExpanderNetTopologyFactory sets up the connections and creates the topo
    """

    def __init__(self, nr_hosts_per_rack, nr_racks, nr_links, tor_switch_config, seed,
                 networkx_wrapper_type=None, host_link_capacity=None):
        """
        Creates the expander-net configuration.

        Args:
            nr_hosts_per_rack: int, number of hosts per rack
            nr_racks: int, number of racks in the topology
            nr_links: degree of the expander graph (nr_links + nr_hosts_per_rack = nr_electrical ports of switch)
            tor_switch_config: AbstractSwitchConfiguration, configuration used to create the tor switches
            seed: seed for RNG for matching generation
            networkx_wrapper_type: class to use as networkx wrapper (defaults to single link wrapper)

        """

        super(ExpanderNetTopologyConfiguration, self).__init__(
            ExpanderNetTopologyFactory.__name__,
            decorated_topology=None
        )

        self.nr_hosts_per_rack = nr_hosts_per_rack
        self.nr_racks = nr_racks
        assert (self.nr_racks % 2) == 0
        self.tor_switch_config = tor_switch_config
        self.seed = seed
        self.nr_links = nr_links
        self.host_link_capacity = host_link_capacity
        if self.host_link_capacity is None:
            self.host_link_capacity = self.tor_switch_config.link_capacity

        self.networkx_wrapper_type = networkx_wrapper_type
        if self.networkx_wrapper_type is None:
            self.networkx_wrapper_type = NetworkXTopologyWrapper

    @property
    def params(self):
        return {
            'nr_hosts_per_rack': self.nr_hosts_per_rack,
            'nr_racks': self.nr_racks,
            'nr_links': self.nr_links,
            'seed': self.seed,
            'tor_switch_config': str(self.tor_switch_config),
            'host_link_capacity': self.host_link_capacity,
            'networkx_wrapper_type': self.networkx_wrapper_type.__name__
        }
