import numpy as np
from dcflowsim import constants
from dcflowsim.network.topology import *
from dcflowsim.network.ocs_topology import *
from dcflowsim.network.network_elements import Host, RotorSwitchConfiguration
from dcflowsim.network.network_elements_factory import produce_switch
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import TopologyInterface
from dcflowsim.utils.factory_decorator import factory_decorator
from .topology_factories import TOPOLOGY_FACTORIES


@factory_decorator(factory_dict=TOPOLOGY_FACTORIES)
class RotorNetTopologyFactory(AbstractTopologyFactory):
    """
    Factory class for RotorNet Topologies.

    Each Rack contains the same number of hosts. A Layer of multiple RotorSwitches connects them (all-to-all).
    """

    @classmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        """
        Factory function. Creates all hosts, switches (electrical and optical) and links.
        Args:
            topology_configuration: RotorNetTopologyConfiguration, config for the RotorNet topology
            decorated_topology: Topology to decorate; must be BaseDataCenterTopology

        Returns: A RotorNet topology; Really a BaseDatacenterTopology decorated with a Topology_with_OCS.

        """

        # make sure we have enough configurations for all rotor switches. If only one config is given, create a
        # list from it to ensure generic creation of the RotorSwitches later on.
        if len(topology_configuration.rotor_switch_configs) != 1:
            assert topology_configuration.nr_rotorswitches == len(topology_configuration.rotor_switch_configs),\
                "Not enough rotorswitch_configs for the requested number of rotorswitches."
            rotor_configs = topology_configuration.rotor_switch_configs
        elif len(topology_configuration.rotor_switch_configs) == 1:
            rotor_configs = [topology_configuration.rotor_switch_configs[0]
                             for _ in range(topology_configuration.nr_rotorswitches)]

        # Create an empty BaseTopo if None is given
        if decorated_topology is None:
            base_dc_topo = BaseDataCenterTopology()
        else:
            assert isinstance(decorated_topology, BaseDataCenterTopology)
            base_dc_topo = decorated_topology

        # Add the ToR switches, hosts and add electrical links between them
        for rack_id in range(topology_configuration.nr_racks):
            tor_switch = produce_switch(
                config=topology_configuration.tor_switch_config,
                node_id="TorSwitch-{}".format(
                    rack_id
                ),
                num_ports=topology_configuration.nr_hosts_per_rack,
                network_identifier="TorSwitch"
            )
            base_dc_topo.add_switch(tor_switch)

            for host_id in range(topology_configuration.nr_hosts_per_rack):
                host = Host(node_id="Host-P{},{}".format(rack_id, host_id))

                base_dc_topo.add_host(host)

                base_dc_topo.add_link(tor_switch,
                                      host,
                                      capacity=topology_configuration.tor_switch_config.link_capacity)

        # create the rotornet rotorswitch connectivity and the
        # ocs_configs for the OCS top config
        base_tor_switches = base_dc_topo.get_switches()
        offset = 0
        rotor_ocs_connectivity = {}
        rotor_switch_configs = {}
        for rotor_switch_id in range(topology_configuration.nr_rotorswitches):
            rotor_ocs_connectivity[rotor_switch_id + offset] = base_tor_switches
            rotor_switch_configs[rotor_switch_id + offset] = rotor_configs[rotor_switch_id]

        # Create the Topo with OCS config, BaseDCTopology has no configuration,
        # therefore decorated_topo_config is None
        ocs_topo_config = TopologyWithOcsConfiguration(
            decorated_topology_configuration=None,
            ocs_connectivity=rotor_ocs_connectivity,
            optical_link_capacity=topology_configuration.optical_link_capacity,
            ocs_configs=rotor_switch_configs)

        # Create RotorNetTopology from the OCSFactory
        rotor_net_topo = TopologyWithOcsFactory.generate_topology(topology_configuration=ocs_topo_config,
                                                                  decorated_topology=base_dc_topo)

        return rotor_net_topo


@peewee_interface_factory_decorator(interface=TopologyInterface)
class RotorNetTopologyConfiguration(AbstractTopologyConfiguration):
    """
    Configuration for a RotorNet topology. RotorNetTopologyFactory sets up the connections and creates the topo
    """
    def __init__(self, nr_hosts_per_rack, nr_racks, tor_switch_config,
                 nr_rotorswitches, rotor_switch_configs, optical_link_capacity,
                 decorated_topology_config=None):
        """
        Creates the Configuration. Uses a BaseDataCenterTopology as decorated_topology.

        Will create nr_rotorswitches RotorSwitches. If rotor_switch_configs is length 1, all RotorSwitches will share
        the same config, if it is on length nr_rotorswitches, the contained configs will be used for the creation.

        Args:
            nr_hosts_per_rack: int, number of hosts per rack
            nr_racks: int, number of racks in the topology
            tor_switch_config: AbstractSwitchConfiguration, configuration used to create the tor switches
            nr_rotorswitches: int, number of rotorswitches in the topology
            rotor_switch_configs: list of AbstractSwitchConfiguration, used to create the single Rotorswitches
            optical_link_capacity: bidirectional capacity of the optical links (assumed to be equal in both directions)
        """

        super(RotorNetTopologyConfiguration, self).__init__(
            RotorNetTopologyFactory.__name__,
            decorated_topology=decorated_topology_config
        )

        self.nr_hosts_per_rack = nr_hosts_per_rack
        self.nr_racks = nr_racks
        self.tor_switch_config = tor_switch_config

        self.nr_rotorswitches = nr_rotorswitches
        self.rotor_switch_configs = rotor_switch_configs
        self.optical_link_capacity = optical_link_capacity

    @property
    def params(self):
        return {
            'nr_hosts_per_rack': self.nr_hosts_per_rack,
            'nr_racks': self.nr_racks,
            'tor_switch_config': str(self.tor_switch_config),
            'nr_rotorswitches': self.nr_rotorswitches,
            'rotor_switch_configs': [str(config) for config in self.rotor_switch_configs],
            'optical_link_capacity': self.optical_link_capacity
        }


def generate_rotorswitch_matching_cycle(nr_ports):
    """
    Generates a matching cycle (nr_ports-1 matchings) for a RotorSwitch using a Round
    Robin Tournament approach.
    See: https://en.wikipedia.org/wiki/Round-robin_tournament#Scheduling_algorithm

    Each matching consists of nr_ports/2 pairings of two ports.
    The whole cycle guarantees that every port is connected to each individual other port a single time.

    One matching has the form [(0, 1), (2, 3)], where (0,1) describes a symmetric connection
    between port 0 and port 1.

    Args:
        nr_ports: int, number of ports the matchings should pair must be even

    Returns:
        list of nr_ports-1 matchings
    """

    assert nr_ports % 2 == 0, "The number of ports must be even but is: {}".format(nr_ports)

    nr_matchings = nr_ports - 1  # Number of matchings for a complete cycle
    nr_cons_per_matching = nr_ports // 2  # Number of connections for a single matching

    matchings_per_step = []
    port_list = np.arange(nr_ports)  # Init the array of ports

    for matching_index in range(nr_matchings):  # loop over the number of needed matchings
        matchings_per_step.append([])

        # loop over the number of connections per matching and pair the ports
        for con in range(nr_cons_per_matching):
            matchings_per_step[matching_index].append((port_list[con], port_list[-con-1]))

        # circshift all ports but the first one to generate the next port list, compare to link in docstring
        port_list[1:] = np.roll(port_list[1:], 1)

    return matchings_per_step


def get_rotorswitch_configs(nr_racks, nr_rotorswitches, link_capacity, night_duration=0,
                            day_duration=constants.ROTORSWITCH_RECONF_PERIOD):
    """
    Returns a list of RotorSwitchConfiguration objects (one for each RotorSwitch in the topology)
    Each switch starts to cycle the same matching list from a different index. Since the reconfiguration times
    and offset are the same for all switches, there is no risk of activating the same matching  multiple times
    in a cycle.

    Args:
        nr_racks: int, number of racks in the topology
        nr_rotorswitches: int, number of rotorswitches
        link_capacity: int, capacity of rotorswitches
        night_duration: int, reconfiguration time of the rotorswitches
        day_duration: int, reschedule period of RotorReconfigEvent

    Returns: list of RotorSwitchConfiguration Objects

    """
    matchings = generate_rotorswitch_matching_cycle(nr_racks)

    # each rotorswitch should start their cycle from somewhere else in the list.
    shift = int(nr_racks / nr_rotorswitches)

    config = list()
    indent = 0
    for _ in range(nr_rotorswitches):
        # append RotorSwitchConfiguration Object with circular shifted matchings so that same matching is
        # active for different RotorSwitches at the same time
        config.append(RotorSwitchConfiguration(matchings[indent:] + matchings[:indent], link_capacity=link_capacity,
                                               day_duration=day_duration, night_duration=night_duration))
        indent += shift  # accumulate the shift

    return config
