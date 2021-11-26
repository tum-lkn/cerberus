from dcflowsim.network import expandernet, ocs_topology, topology, rotornet, network_elements
from dcflowsim.network.networkx_wrapper import NetworkXMultiGraphTopologyWrapper
from dcflowsim.network.topology import AbstractTopologyFactory, AbstractTopologyConfiguration
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import TopologyInterface
from dcflowsim.utils.factory_decorator import factory_decorator
from .topology_factories import TOPOLOGY_FACTORIES


@factory_decorator(factory_dict=TOPOLOGY_FACTORIES)
class CerberusTopologyFactory(AbstractTopologyFactory):
    """
    Factory for Cerberus topology. Produces a three tiered topology with expander, RotorNet and Demand-aware (OCS) part
    """

    @classmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        assert isinstance(topology_configuration, CerberusTopologyConfiguration)
        # First build the ExpanderNet topology
        dec_topology = expandernet.ExpanderNetTopologyFactory.generate_topology(
            topology_configuration.expander_config
        )

        # Then build the OCS/Rotor parts
        return ocs_topology.TopologyWithOcsFactory.generate_topology(
            topology_configuration.ocs_topo_config,
            decorated_topology=dec_topology
        )


@peewee_interface_factory_decorator(interface=TopologyInterface)
class CerberusTopologyConfiguration(AbstractTopologyConfiguration):
    """
    Configuration for a Cerberus topology. CerberusTopologyFactory sets up the connections and creates the topo
    """

    def __init__(self, nr_hosts_per_rack, nr_racks, nr_static, nr_rotor, nr_da, link_capacity, seed, rotor_day_duration,
                 rotor_night_duration, ocs_reconfig_delay, tor_switch_config):
        """
        Creates configuration for Cerberus topology (static topology is currently fixed to expander). Parameters
        match to those of the individual topologies
        Args:
            nr_hosts_per_rack: Number of hosts per rack
            nr_racks: Number of racks
            nr_static: Number of static switches (degree of expander topology)
            nr_rotor: Number of Rotor switches
            nr_da: Number of OCS switches
            link_capacity: Capacity of single link (equal for all topologies)
            seed: Seed for expander
            rotor_day_duration: RotorNet Reconfig Event period
            rotor_night_duration: RotorNet reconfiguration delay
            ocs_reconfig_delay: OCS reconfiguration delay
            tor_switch_config: Instance of ElectricalWithOpticalSwitchConfig describing the ToRs
        """

        super(CerberusTopologyConfiguration, self).__init__(
            CerberusTopologyFactory.__name__,
            decorated_topology=None
        )

        self.nr_hosts_per_rack = nr_hosts_per_rack
        assert self.nr_hosts_per_rack == 1, "Currently only Super Source is supported"
        self.nr_racks = nr_racks
        assert (self.nr_racks % 2) == 0
        self.link_capacity = link_capacity
        self.tor_switch_config = tor_switch_config
        assert isinstance(self.tor_switch_config, network_elements.ElectricalSwitchWithOpticalPortsConfiguration)
        assert self.link_capacity == self.tor_switch_config.link_capacity

        # Static/expander parameter
        self.nr_static = nr_static
        self.seed = seed

        # Rotornet parameter
        self.nr_rotor = nr_rotor
        self.rotor_day_duration = rotor_day_duration
        self.rotor_night_duration = rotor_night_duration

        # Demand-aware paramter
        self.nr_da = nr_da
        self.ocs_reconfig_delay = ocs_reconfig_delay

        # Always use MultiGraphWrapper here
        self.networkx_wrapper_type = NetworkXMultiGraphTopologyWrapper

        assert self.nr_rotor + self.nr_static + self.nr_da <= self.tor_switch_config.num_optical_ports

        self.expander_config = None
        self.rotorswitch_configs = None
        self.ocs_topo_config = None
        self.__build_topo_configs()

    def __build_topo_configs(self):
        """
        Build concrete topology configs from given parameters
        Returns:

        """
        self.expander_config = expandernet.ExpanderNetTopologyConfiguration(
            nr_hosts_per_rack=self.nr_hosts_per_rack,
            nr_racks=self.nr_racks,
            nr_links=self.nr_static,
            tor_switch_config=self.tor_switch_config,
            seed=self.seed,
            networkx_wrapper_type=self.networkx_wrapper_type,
            # Multiply link_capacity with number of up-links if we have only one host per Rack (Super-source)
            host_link_capacity=self.link_capacity * (
                (self.nr_static + self.nr_rotor + self.nr_da) if self.nr_hosts_per_rack == 1 else 1
            )
        )

        self.rotorswitch_configs = {
            i: rconfig for i, rconfig in enumerate(
                rotornet.get_rotorswitch_configs(nr_racks=self.nr_racks, nr_rotorswitches=self.nr_rotor,
                                                 link_capacity=self.link_capacity,
                                                 night_duration=self.rotor_night_duration,
                                                 day_duration=self.rotor_day_duration)
            )
        }

        # Add DA-switch configs
        ocs_configs = {
            i: network_elements.OpticalCircuitSwitchConfiguration(
                link_capacity=self.link_capacity,
                reconfig_delay=self.ocs_reconfig_delay
            ) for i in
            range(len(self.rotorswitch_configs), len(self.rotorswitch_configs) + self.nr_da)
        }
        ocs_configs.update(self.rotorswitch_configs)
        self.ocs_topo_config = ocs_topology.TopologyWithOcsConfiguration(
            decorated_topology_configuration=self.expander_config,
            # Connect all CSs to ToRs
            ocs_connectivity={i: topology.KNOWN_SWITCH_TYPE_TOR_SWITCH for i in range(len(ocs_configs))},
            optical_link_capacity=self.link_capacity,
            ocs_configs=ocs_configs

        )

    @property
    def params(self):
        return {
            'nr_hosts_per_rack': self.nr_hosts_per_rack,
            'nr_racks': self.nr_racks,
            'tor_switch_config': str(self.tor_switch_config),
            'link_capacity': self.link_capacity,
            'networkx_wrapper_type': self.networkx_wrapper_type.__name__,
            'nr_static': self.nr_static,
            'seed': self.seed,
            'nr_rotor': self.nr_rotor,
            'rotor_day_duration': self.rotor_day_duration,
            'rotor_night_duration': self.rotor_night_duration,
            'nr_da': self.nr_da,
            'ocs_reconfig_delay': self.ocs_reconfig_delay
        }
