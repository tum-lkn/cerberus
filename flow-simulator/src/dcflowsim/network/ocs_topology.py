import itertools
import logging
import copy
from decimal import Decimal

from dcflowsim import literals
from dcflowsim.network.network_elements import OpticalCircuitSwitchConfiguration, RotorSwitchConfiguration
from dcflowsim.network.network_elements_factory import produce_switch
from dcflowsim.network.networkx_wrapper import NetworkXMultiGraphTopologyWrapper
from . import topology, network_elements
from dcflowsim import configuration
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import TopologyInterface
from dcflowsim.utils.factory_decorator import factory_decorator
from .topology_factories import TOPOLOGY_FACTORIES, TOPOLOGY_VIEW_FACTORIES


class TopologyWithOcs(topology.TopologyDecorator):
    """
    Decorator for topologies that contain Optical Circuit Switches.
    """

    def __init__(self, decorated_topology, topology_type="TopologyWithOcs", track_flows_on_links=True):
        super(TopologyWithOcs, self).__init__(decorated_topology, topology_type)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__topology_type = topology_type
        self.__track_flows_on_links = track_flows_on_links
        # OCS's and links from/to OCS are not added to decorated topology. We must save them here
        self.__ocs = list()
        self._logical_link_to_ocs_map = network_elements.MultiLinkMapper()
        # Used to update the views if optical links are changed
        self.__views = list()
        # if the NetworkXWrapper is a MultiGraph one, we will need MultiLinkcontainers:
        if decorated_topology is not None \
                and isinstance(decorated_topology.network_graph,
                               NetworkXMultiGraphTopologyWrapper):
            self.__ocs_links = network_elements.MultiLinkContainer()
            self.__logical_links = network_elements.MultiLinkContainer()
        else:
            self.__ocs_links = network_elements.LinkContainer()
            self.__logical_links = network_elements.LinkContainer()

    @property
    def views(self):
        return self.__views

    @property
    def network_graph(self):
        return self.decorated_topology.network_graph

    @property
    def links(self):
        """ Returns a list of actual link objects to iterate over"""
        # combine the data of both link containers and create a new one of same type as
        # decorated topo
        return self.decorated_topology.links.__class__(
            {**self.decorated_topology.links, **self.__ocs_links})

    @property
    def ocs(self):
        return self.__ocs

    @property
    def ocs_links(self):
        return self.__ocs_links

    @property
    def logical_links(self):
        return self.__logical_links

    @property
    def logical_link_to_ocs_map(self):
        return self._logical_link_to_ocs_map

    @topology.update_views
    def add_node(self, node, **kwargs):
        if isinstance(node, network_elements.OpticalCircuitSwitch):
            if node in self.ocs:
                raise RuntimeError("OCS already added to topology.")
            self.ocs.append(node)
        else:
            # No OCS node, forward to decorated topology
            self.decorated_topology.add_node(node, **kwargs)

    def get_node(self, node_id):
        for ocs in self.ocs:
            if node_id == ocs.node_id:
                return ocs
        return self.decorated_topology.get_node(node_id)

    @topology.update_views
    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        if node_1 in self.ocs:
            link = network_elements.OpticalLink(
                node_1, node_2, kwargs["capacity"], track_flows=self.__track_flows_on_links
            )
            node_1.add_link(link)
            node_2.add_link(link)

            # This link is not added to network graph. So save a handle here
            self.__ocs_links[(node_1.node_id, node_2.node_id, link.link_identifier)] = link
        elif node_2 in self.ocs:
            link = network_elements.OpticalLink(
                node_2, node_1, kwargs["capacity"], track_flows=self.__track_flows_on_links
            )
            node_2.add_link(link)
            node_1.add_link(link)
            # This link is not added to network graph. So save a handle here
            self.__ocs_links[(node_1.node_id, node_2.node_id, link.link_identifier)] = link
        else:
            # Link is not related to OCS. Forward to decorated topology
            self.decorated_topology.add_link(node_1, node_2, link_identifier, **kwargs)

    @topology.update_views
    def remove_link(self, node_1, node_2, link_identifier=None):
        if node_1 in self.ocs:
            node_1.remove_link(self.__ocs_links[(node_1.node_id, node_2.node_id)])
            node_2.remove_link(self.__ocs_links[(node_1.node_id, node_2.node_id)])
            del self.__ocs_links[(node_1.node_id, node_2.node_id, link_identifier)]
        elif node_2 in self.ocs:
            node_2.remove_link(self.__ocs_links[(node_2.node_id, node_1.node_id)])
            node_1.remove_link(self.__ocs_links[(node_2.node_id, node_1.node_id)])
            del self.__ocs_links[(node_1.node_id, node_2.node_id, link_identifier)]
        else:
            # Link is not related to OCS. Forward to decorated topology
            self.decorated_topology.remove_link(node_1, node_2, link_identifier)

    def get_link(self, node_1, node_2, link_identifier=None):
        """
        Returns the link object of the link between node_1 and node_2
        Forwards the call to decorated topology
        Args:
            node_1: either instance of Node or the node id (int)
            node_2: either instance of Node or the node id (int)
            link_identifier: Link identifier for the link

        Returns:
            Link object
        Raises:
            RuntimeError if there is no link between node_1 and node_2
        """
        try:
            return self.__logical_links[(node_1, node_2, link_identifier)]
        except KeyError:
            return self.decorated_topology.get_link(node_1, node_2, link_identifier)

    def get_network_graph(self):
        return self.decorated_topology.get_network_graph()

    def get_nodes_by_type(self, node_type):
        # Filter for potential ocs first
        if node_type in OCSSwitchTypes.list:
            return [ocs for ocs in self.ocs if ocs.network_identifier == node_type]
        # Else propagate to decorated
        return self.decorated_topology.get_nodes_by_type(node_type)

    def get_nodes(self):
        return self.decorated_topology.get_nodes() + list(self.__ocs)

    def get_links(self):
        return self.decorated_topology.get_links()

    def get_hosts(self):
        return self.decorated_topology.get_hosts()

    def get_shortest_path(self, source, destination, weight):
        return self.decorated_topology.get_shortest_path(
            source,
            destination,
            weight
        )

    @staticmethod
    def link_identifier_with_node(node, linkidentifier_basic):
        return f"{linkidentifier_basic}-{node.node_id}"

    @topology.update_views
    def set_circuit(self, ocs, node_1, node_2):
        """
        Set a circuit on the OCS. (Logically) connect nodes and add edge to decorated topology
        Args:
            ocs (OpticalCircuitSwitch):
            node_1 (Node):
            node_2 (Node):

        Returns:

        """
        assert isinstance(ocs, network_elements.OpticalCircuitSwitch)
        if ocs not in self.ocs:
            raise RuntimeError("OCS does not belong to this topology.")
        if node_1.node_id not in ocs.physical_neighbors.keys():
            raise RuntimeError("Node1 has no wire to OCS.")
        if node_2.node_id not in ocs.physical_neighbors.keys():
            raise RuntimeError("Node2 has no wire to OCS.")
        ocs.set_circuit(node_1.node_id, node_2.node_id)
        node_1.connect(node_2, self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic))
        node_2.connect(node_1, self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic))

        opt_capacity = min(
            self.__ocs_links[(node_1.node_id, ocs.node_id)].total_capacity,
            self.__ocs_links[(node_2.node_id, ocs.node_id)].total_capacity
        )
        self.__logical_links[(node_1.node_id, node_2.node_id)] = \
            network_elements.OpticalLink(
                node_1, node_2, opt_capacity,
                link_identifier=self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic),
                track_flows=self.__track_flows_on_links
            )
        # Safe a mapping between the logical link and the corresponding ocs
        self._logical_link_to_ocs_map[(node_1.node_id, node_2.node_id,
                                       self.link_identifier_with_node(ocs,
                                                                      network_elements.LinkIdentifiers.dynamic))] = ocs.node_id

        # Update logical representation, i.e., the network graph
        self.decorated_topology.network_graph.add_link(
            node_1,
            node_2,
            link_identifier=self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic),
            capacity=opt_capacity,
            remaining_capacity=opt_capacity
        )
        return opt_capacity

    @topology.update_views
    def reset_circuit(self, ocs, node1, node2):
        """
        Reset/Remove circuit. (logically) disconnect nodes and remove edge from decorated topology
        Args:
            ocs (OpticalCircuitSwitch):
            node1 (Node):
            node2 (Node):

        Returns:

        """
        assert isinstance(ocs, network_elements.OpticalCircuitSwitch)
        if ocs not in self.ocs:
            raise RuntimeError("OCS does not belong to this topology.")
        if node1.node_id not in ocs.physical_neighbors:
            raise RuntimeError("Node1 has no wire to OCS.")
        if node2.node_id not in ocs.physical_neighbors:
            raise RuntimeError("Node2 has no wire to OCS.")
        ocs.release_circuit(node1.node_id, node2.node_id)
        node1.disconnect(node2, self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic))
        node2.disconnect(node1, self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic))

        # Get flows on this link and release them one by one
        flows_to_release = self.__logical_links[(node1.node_id, node2.node_id,
                                                 self.link_identifier_with_node(ocs,
                                                                                network_elements.LinkIdentifiers.dynamic))].flows
        while len(flows_to_release):
            f = flows_to_release[0]
            # so we have a new first element. Idea is to avoid to copy list
            self.release_route(f)
            f.reset()

        # Remove logical link and mapping between logical link and ocs
        del self.__logical_links[(node1.node_id, node2.node_id,
                                  self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic))]
        del self._logical_link_to_ocs_map[(node1.node_id, node2.node_id,
                                           self.link_identifier_with_node(ocs,
                                                                          network_elements.LinkIdentifiers.dynamic))]

        # Update logical representation, i.e., the network graph
        self.decorated_topology.network_graph.remove_link(
            node1,
            node2,
            link_identifier=self.link_identifier_with_node(ocs, network_elements.LinkIdentifiers.dynamic)
        )
        self.logger.debug("Reset circuit {}-{}".format(node1, node2))

    @topology.update_views
    def reset_all_circuits(self):
        for ocs in self.ocs:
            for node1, node2 in itertools.product(ocs.physical_neighbors.values(),
                                                  ocs.physical_neighbors.values()):
                if node1 == node2:
                    continue
                try:
                    self.reset_circuit(ocs, node1, node2)
                except RuntimeError:
                    continue

    @topology.update_views
    def allocate_route(self, flow):
        # Use tuple unpacking to deal with standard links (src, dest) and (src, dest, id)
        for link_tuple in flow.network_route.links:
            self.get_link(*link_tuple).allocate(flow)
        self.decorated_topology.network_graph.allocate_route(flow.network_route)

    @topology.update_views
    def release_route(self, flow):
        for link_tuple in flow.network_route.links:
            self.get_link(*link_tuple).release(flow)
        self.network_graph.release_route(flow.network_route)

    @topology.update_views
    def reset(self):
        self.reset_all_circuits()
        self.disconnect_all_rotorswitches()
        self.reset_rotorswitches()
        self.decorated_topology.reset()
        self.connect_all_rotorswitches()

    def get_corresponding_tor_switch(self, node):
        return self.decorated_topology.get_corresponding_tor_switch(node)

    def connect_node_pair_via_rotorswitch(self, rotor, node1_id, node2_id):
        node_1 = self.get_node(node1_id)
        node_2 = self.get_node(node2_id)
        if node_1.node_id not in rotor.physical_neighbors.keys():
            raise RuntimeError("Node1 has no wire to OCS.")
        if node_2.node_id not in rotor.physical_neighbors.keys():
            raise RuntimeError("Node2 has no wire to OCS.")

        node_1.connect(node_2, network_elements.LinkIdentifiers.rotor)
        node_2.connect(node_1, network_elements.LinkIdentifiers.rotor)

        opt_capacity = min(
            self.__ocs_links[(node_1.node_id, rotor.node_id)].total_capacity,
            self.__ocs_links[(node_2.node_id, rotor.node_id)].total_capacity
        )
        self.__logical_links[(node_1.node_id, node_2.node_id)] = network_elements.OpticalLink(
            node_1, node_2, opt_capacity,
            link_identifier=network_elements.LinkIdentifiers.rotor,
            track_flows=self.__track_flows_on_links
        )
        # Safe a mapping between the logical link and the corresponding ocs
        self._logical_link_to_ocs_map[(node_1.node_id, node_2.node_id,
                                       network_elements.LinkIdentifiers.rotor)] = rotor.node_id

        # Update logical representation, i.e., the network graph
        self.decorated_topology.network_graph.add_link(
            node_1,
            node_2,
            link_identifier=network_elements.LinkIdentifiers.rotor,
            capacity=opt_capacity,
            remaining_capacity=opt_capacity
        )
        self.logger.debug("Connected nodes {}-{}".format(node_1, node_2))

    @topology.update_views
    def connect_nodes_via_rotorswitch(self, rotor, new_matchings):
        """
        Sets the logical representation of the circuit. Creates the logical links. Currently, only used for RotorNet
        Topology.
        Args:
            rotor: object of type RotorSwitch
            new_matchings: a list of tuples [(node1_id, node2_id), ...] containing the new matchings

        Returns: optical capacity of the connection

        """
        assert isinstance(rotor, network_elements.RotorSwitch)
        if rotor not in self.ocs:
            raise RuntimeError(" does not belong to this topology.")

        for node1_id, node2_id in new_matchings:
            self.connect_node_pair_via_rotorswitch(rotor, node1_id, node2_id)

    def disconnect_node_pair_via_rotorswitch(self, rotor, node1_id, node2_id):
        node_1 = self.get_node(node1_id)
        node_2 = self.get_node(node2_id)
        if node_1.node_id not in rotor.physical_neighbors:
            raise RuntimeError("Node1 has no wire to OCS.")
        if node_2.node_id not in rotor.physical_neighbors:
            raise RuntimeError("Node2 has no wire to OCS.")

        node_1.disconnect(node_2, network_elements.LinkIdentifiers.rotor)
        node_2.disconnect(node_1, network_elements.LinkIdentifiers.rotor)

        # Get flows on this link and release them one by one
        flows_to_release = self.__logical_links[(node_1.node_id, node_2.node_id,
                                                 network_elements.LinkIdentifiers.rotor)].flows
        while len(flows_to_release):
            f = flows_to_release[0]
            # so we have a new first element. Idea is to avoid to copy list
            self.release_route(f)
            f.reset()

        # Remove the logical link and the mapping between it and the rotor
        del self.__logical_links[(node_1.node_id, node_2.node_id,
                                  network_elements.LinkIdentifiers.rotor)]
        del self._logical_link_to_ocs_map[(node_1.node_id, node_2.node_id,
                                           network_elements.LinkIdentifiers.rotor)]

        # Update logical representation, i.e., the network graph
        self.decorated_topology.network_graph.remove_link(
            node_1,
            node_2,
            link_identifier=network_elements.LinkIdentifiers.rotor
        )
        self.logger.debug("Disconnected Nodes {}-{}".format(node_1, node_2))

    @topology.update_views
    def disconnect_nodes_via_rotorswitch(self, rotor, matchings_to_disconnect):
        """
        Removes the logical representation of the circuit. Deletes the logical links. Currently, only being used for
        RotorNet Topology.
        Args:
            rotor: object of type RotorSwitch
            matchings_to_disconnect: a list of tuples [(node1_id, node2_id), ...] to be disconnected

        Returns: Nothing

        """

        assert type(rotor) is network_elements.RotorSwitch
        if rotor not in self.ocs:
            raise RuntimeError("OCS does not belong to this topology.")
        for node1_id, node2_id in matchings_to_disconnect:
            self.disconnect_node_pair_via_rotorswitch(rotor, node1_id, node2_id)

    @topology.update_views
    def connect_rotor_matching(self, rotor):
        return rotor.connect_matching(self)

    @topology.update_views
    def disconnect_rotor_matching(self, rotor):
        return rotor.disconnect_matching(self)

    def disconnect_all_rotorswitches(self):
        for rotorswitch in self.ocs:
            if type(rotorswitch) == network_elements.RotorSwitch:
                self.disconnect_rotor_matching(rotorswitch)

    def connect_all_rotorswitches(self):
        self.reset_rotorswitches()
        for rotorswitch in self.ocs:
            if type(rotorswitch) == network_elements.RotorSwitch:
                self.connect_rotor_matching(rotorswitch)

    def reset_rotorswitches(self):
        for rotorswitch in self.ocs:
            if type(rotorswitch) == network_elements.RotorSwitch:
                rotorswitch.reset()

    def get_ocs_for_logical_link(self, node_id_1, node_id_2, link_identifier):
        """ Retrieves the OCS that 'creates' a specific logical link

        Args:
            node_id_1: NodeID of first node
            node_id_2: NodeID of second node
            link_identifier: Link identifier of the
        """
        return self._logical_link_to_ocs_map[(node_id_1, node_id_2, link_identifier)]

    def add_view(self, topology_view):
        if topology_view is self or topology_view in self.views:
            return
        self.views.append(topology_view)


@peewee_interface_factory_decorator(interface=TopologyInterface)
class TopologyWithOcsConfiguration(topology.AbstractTopologyConfiguration):
    def __init__(self, decorated_topology_configuration, ocs_connectivity, optical_link_capacity, ocs_configs=None):
        """
        Config for Topologies containing OCS. cos_configs can be used to determine what kind of OCS will be used.
        If ocs_configs is None, standard OCS will be created according to the ocs_connectivity.
        By Passing a dict containing the same keys as the ocs_connectivity dict as ocs_configs, the OCS will be created
        according to the configs. (Currently only used to create RotorNet)

        Args:
            decorated_topology_configuration: instance of AbstractTopologyConfiguration

            ocs_connectivity: dict describing how the ocs are connected to the other nodes.
                keys are ocs id offset to existing, values are list of node_ids that shall be connected.
                Implicitly gives the number of OCSs and their number of ports.
                Special Cases   - Fat Tree: value can be a string that indicates which
                                level of switches of the Fattree to connect
                                - BaseDCTopo: value can be 'TorSwitch' to connect to all
                                nodes in the decorated topo with this network_identifier

            ocs_configs: optional dict, ocs id -> RotorSwitch or OCS configs, can be used
                        for creation of mixed OCS topologies.
                        If None, standard OCS will be created
        """
        super(TopologyWithOcsConfiguration, self).__init__(
            TopologyWithOcsFactory.__name__, decorated_topology_configuration)
        self.ocs_connectivity = ocs_connectivity
        self.optical_link_capacity = Decimal(optical_link_capacity)

        # No OCS configs are given, assume standard OCS should be created
        if ocs_configs is None:
            self.ocs_configs = {ocs_number: OpticalCircuitSwitchConfiguration(optical_link_capacity)
                                for ocs_number in ocs_connectivity.keys()}
        else:
            self.ocs_configs = ocs_configs

    @property
    def params(self):
        return {
            'ocs_connectivity': self.ocs_connectivity,
            'opt_link_capacity': self.optical_link_capacity,
            'ocs_configs': [(str(config), str(param)) for (config, param) in zip(self.ocs_configs.keys(),
                                                                                 self.ocs_configs.values())]
        }


@factory_decorator(factory_dict=TOPOLOGY_FACTORIES)
class TopologyWithOcsFactory(topology.AbstractTopologyFactory):
    @classmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        """

        Args:
            topology_configuration: instance of AbstractTopologyConfiguration
            decorated_topology:

        Returns:

        """
        if decorated_topology is None:
            decorated_topology = topology.BaseDataCenterTopology()

        ocs_topo = TopologyWithOcs(decorated_topology)
        assert len(topology_configuration.ocs_configs) == \
               len(topology_configuration.ocs_connectivity), "Number of connectivity and config items should match"

        for ocs_node_number, attached_nodes in \
                topology_configuration.ocs_connectivity.items():

            if type(attached_nodes) is str:
                # Attatching an OCS/Rotor to a certain level in the Topology is
                # supported for FatTree (Edge, Aggr, Core) and for BaseDCTopo (TorSwitch)
                if isinstance(decorated_topology, topology.BaseDataCenterTopology):
                    attached_nodes = decorated_topology.get_nodes_by_type(attached_nodes)
                else:
                    raise ValueError("OCS connectivity to a certain level is only "
                                     "supported for Fattree and BaseDCTopo")
            # Get number of ports
            num_ports = len(attached_nodes)
            assert num_ports > 0, "Creation of OCS with zero ports, " \
                                  "please check ocs_connectivity settings."

            ocs_config = topology_configuration.ocs_configs[ocs_node_number]

            if isinstance(ocs_config, OpticalCircuitSwitchConfiguration):
                node_id = "OCS-{}".format(ocs_node_number)
                network_identifier = OCSSwitchTypes.ocs
            elif isinstance(ocs_config, RotorSwitchConfiguration):
                node_id = "RotorSwitch-{}".format(ocs_node_number)
                network_identifier = OCSSwitchTypes.rotor
            else:
                raise RuntimeError("OCS config must be either "
                                   "OpticalCircuitSwitchConfiguration or "
                                   "RotorSwitchConfiguration")

            ocs = produce_switch(ocs_config, node_id, num_ports, network_identifier)

            ocs_topo.add_node(ocs)

            # Connect OCS with other nodes
            for node in attached_nodes:
                if not isinstance(node, network_elements.Node):
                    node = decorated_topology.get_node(node)
                ocs_topo.add_link(ocs, node, capacity=topology_configuration.optical_link_capacity)

        # Rotors need to set up the first matching
        for rotor_switch in ocs_topo.get_nodes_by_type(OCSSwitchTypes.rotor):
            rotor_switch.connect_matching(ocs_topo)

        return ocs_topo


class TopologyWithOcsView(topology.AbstractTopology):
    def __init__(self, topology_copy, relevant_link_identifier, relevant_ocs_identifier):
        assert isinstance(topology_copy, TopologyWithOcs)
        self.__topology_copy = topology_copy
        self.__relevant_link_identifier = relevant_link_identifier
        self.__relevant_ocs_identifier = relevant_ocs_identifier

    @property
    def network_graph(self):
        return self.__topology_copy.network_graph

    @property
    def links(self):
        return self.__topology_copy.links

    @property
    def ocs(self):
        return self.__topology_copy.ocs

    @property
    def ocs_links(self):
        return self.__topology_copy.ocs_links

    @property
    def logical_links(self):
        return self.__topology_copy.logical_links

    @property
    def logical_link_to_ocs_map(self):
        return self.__topology_copy.logical_link_to_ocs_map

    def add_node(self, node, **kwargs):
        node_copy = copy.deepcopy(node)
        self.__topology_copy.add_node(node_copy, **kwargs)

    def get_node(self, node_id):
        return self.__topology_copy.get_node(node_id)

    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        if link_identifier.split("-")[0] not in self.__relevant_link_identifier:
            return
        node_1 = self.__topology_copy.get_node(node_1.node_id)
        node_2 = self.__topology_copy.get_node(node_2.node_id)
        self.__topology_copy.add_link(node_1, node_2, link_identifier, **kwargs)

    def remove_link(self, node_1, node_2, link_identifier=None):
        if link_identifier.split("-")[0] not in self.__relevant_link_identifier:
            return
        node_1 = self.__topology_copy.get_node(node_1.node_id)
        node_2 = self.__topology_copy.get_node(node_2.node_id)
        self.__topology_copy.remove_link(node_1, node_2, link_identifier)

    def get_link(self, node_1, node_2, link_identifier=None):
        if link_identifier.split("-")[0] not in self.__relevant_link_identifier:
            return
        return self.__topology_copy.get_link(node_1, node_2, link_identifier)

    def get_network_graph(self):
        return self.__topology_copy.get_network_graph()

    def get_nodes(self):
        return self.__topology_copy.get_nodes()

    def get_nodes_by_type(self, node_type):
        return self.__topology_copy.get_nodes_by_type(node_type)

    def get_hosts(self):
        return self.__topology_copy.get_hosts()

    def get_links(self):
        return self.__topology_copy.get_links()

    def get_ocs_for_logical_link(self, node_id_1, node_id_2, link_identifier):
        return self.__topology_copy.get_ocs_for_logical_link(node_id_1, node_id_2, link_identifier)

    def get_shortest_path(self, source, destination, weight):
        source = self.__topology_copy.get_node(source.node_id)
        destination = self.__topology_copy.get_node(destination.node_id)
        return self.__topology_copy.get_shortest_path(source, destination, weight)

    def allocate_route(self, flow):
        """
        Allocate the network route of flow on all links that are part of the view. Important for Host links!!
        Args:
            flow: instance of flow

        Returns:

        """
        for link_tuple in flow.network_route.links:
            if link_tuple not in self.links and link_tuple not in self.logical_links:
                continue
            self.get_link(*link_tuple).allocate(flow)

            self.network_graph.get_link(link_tuple[0], link_tuple[1], link_tuple[2])[literals.LINK_REMAINING_CAPACITY] \
                = self.network_graph.get_link(link_tuple[0], link_tuple[1], link_tuple[2])[
                      literals.LINK_REMAINING_CAPACITY] - flow.network_route.rate

            if self.network_graph.get_link(link_tuple[0], link_tuple[1], link_tuple[2])[
                literals.LINK_REMAINING_CAPACITY] < 0:
                raise RuntimeError("Remaining capacity should not be negative")

    def release_route(self, flow):
        for link_tuple in flow.network_route.links:
            if link_tuple not in self.links and link_tuple not in self.logical_links:
                continue
            self.get_link(*link_tuple).release(flow)
            self.network_graph.get_link(link_tuple[0],
                                        link_tuple[1],
                                        link_tuple[2])[literals.LINK_REMAINING_CAPACITY] += flow.network_route.rate
            if self.network_graph.get_link(link_tuple[0],
                                           link_tuple[1],
                                           link_tuple[2])[literals.LINK_REMAINING_CAPACITY] > \
                    self.network_graph.get_link(link_tuple[0],
                                                link_tuple[1],
                                                link_tuple[2])[literals.LINK_CAPACITY]:
                raise RuntimeError(
                    "Remaining capacity should not be larger than total capacity")

    def reset(self):
        for link_tuple in self.get_links():
            self.get_link(*link_tuple).reset()
            self.network_graph.reset_link(*link_tuple)

    def get_corresponding_tor_switch(self, node):
        node = self.__topology_copy.get_node(node.node_id)
        self.__topology_copy.get_corresponding_tor_switch(node)

    def set_circuit(self, ocs, node_1, node_2):
        try:
            ocs = self.__topology_copy.get_node(ocs.node_id)
            node_1 = self.__topology_copy.get_node(node_1.node_id)
            node_2 = self.__topology_copy.get_node(node_2.node_id)
            self.__topology_copy.set_circuit(ocs, node_1, node_2)
        except KeyError:
            pass

    def reset_circuit(self, ocs, node1, node2):
        try:
            ocs = self.__topology_copy.get_node(ocs.node_id)
            node_1 = self.__topology_copy.get_node(node1.node_id)
            node_2 = self.__topology_copy.get_node(node2.node_id)
            self.__topology_copy.reset_circuit(ocs, node_1, node_2)
        except KeyError:
            # One of the nodes is not part of the topology. skip
            pass

    def reset_all_circuits(self):
        self.__topology_copy.reset_all_circuits()

    def __filter_matchings(self, matchings):
        """
        Iterates over matching and only returns those which have both nodes in the topology view.
        KeyErrors are catched and not propagated
        Args:
            matchings:

        Returns:

        """
        filtered_matchings = list()
        for node_1, node_2 in matchings:
            try:
                # Try to get nodes from copy as a check if the view has them.
                node_1 = self.__topology_copy.get_node(node_1)
                node_2 = self.__topology_copy.get_node(node_2)
                # Matchings are still given by node ids
                filtered_matchings.append((node_1.node_id, node_2.node_id))
            except KeyError:
                continue
        return filtered_matchings

    def connect_nodes_via_rotorswitch(self, rotor, new_matchings):
        try:
            ocs = self.__topology_copy.get_node(rotor.node_id)
            filtered_matchings = self.__filter_matchings(new_matchings)
            self.__topology_copy.connect_nodes_via_rotorswitch(ocs, filtered_matchings)
        except KeyError:
            # One of the nodes is not part of the topology. skip
            pass

    def disconnect_nodes_via_rotorswitch(self, rotor, matchings_to_disconnect):
        try:
            ocs = self.__topology_copy.get_node(rotor.node_id)
            filtered_matchings = self.__filter_matchings(matchings_to_disconnect)
            self.__topology_copy.disconnect_nodes_via_rotorswitch(ocs, filtered_matchings)
        except KeyError:
            # One of the nodes is not part of the topology. skip
            pass

    def disconnect_rotor_matching(self, rotor):
        try:
            rotor = self.__topology_copy.get_node(rotor.node_id)
            return self.__topology_copy.disconnect_rotor_matching(rotor)
        except KeyError:
            return

    def connect_rotor_matching(self, rotor):
        try:
            rotor = self.__topology_copy.get_node(rotor.node_id)
            return self.__topology_copy.connect_rotor_matching(rotor)
        except KeyError:
            return

    def add_view(self, topology_view):
        pass


class TopologyWithOcsViewConfiguration(configuration.AbstractConfiguration):
    def __init__(self, relevant_link_identifiers, relevant_ocs_identifiers, decorated_view_configuration=None):
        """

        Args:
            relevant_link_identifiers: list of link identifiers to use
        """
        super(TopologyWithOcsViewConfiguration, self).__init__(TopologyWithOcsViewFactory.__name__)
        self.relevant_link_identifiers = relevant_link_identifiers
        self.relevant_ocs_identifiers = relevant_ocs_identifiers

        self.decorated_topology_view_configuration = decorated_view_configuration
        if self.decorated_topology_view_configuration is None:
            self.decorated_topology_view_configuration = topology.TopologyViewConfiguration(
                relevant_link_identifiers=self.relevant_link_identifiers
            )

    @property
    def params(self):
        return {
            'relevant_link_identifiers': self.relevant_link_identifiers,
            'relevant_ocs_identifiers': self.relevant_ocs_identifiers,
            'decorated_topology_view': self.decorated_topology_view_configuration.params
        }


@factory_decorator(factory_dict=TOPOLOGY_VIEW_FACTORIES)
class TopologyWithOcsViewFactory(object):
    @classmethod
    def produce(cls, topology_view_configuration, original_topology):
        from dcflowsim.network import topology_factories
        assert isinstance(original_topology, TopologyWithOcs)
        assert isinstance(topology_view_configuration, TopologyWithOcsViewConfiguration)

        # Create copy of OCS topology with base topology already filled
        topo_view = TopologyWithOcs(
            decorated_topology=topology_factories.produce_view(
                topology_view_configuration.decorated_topology_view_configuration, original_topology.decorated_topology
            ),
            track_flows_on_links=False
        )

        # Add copies of OCSs and corresponding links
        for ocs in original_topology.ocs:
            if ocs.__class__.__name__ in topology_view_configuration.relevant_ocs_identifiers:
                topo_view.add_node(ocs.copy())

        for (node_id_1, node_id_2, link_identifier), link_object in original_topology.ocs_links.items():
            try:
                node_1 = topo_view.get_node(node_id_1)
                node_2 = topo_view.get_node(node_id_2)
            except KeyError:
                # At least one of the ndoes is not part of the topology
                continue

            topo_view.add_link(
                node_1, node_2, link_identifier, capacity=link_object.total_capacity
            )

        # Add copies of set logical links if they are in our view
        for (node1_id, node2_id, link_identifier), link_object in original_topology.logical_links.items():
            # Use in View if in filtered network graph
            if (node1_id, node2_id, link_identifier) in topo_view.network_graph.get_links():
                node1 = topo_view.get_node(node1_id)
                node2 = topo_view.get_node(node2_id)
                topo_view.logical_links[(node1_id, node2_id, link_identifier)] = network_elements.OpticalLink(
                    node1, node2,
                    link_object.total_capacity,
                    link_identifier=link_identifier,
                    track_flows=False
                )

                # If necessary add OCSs that are used by these links but not yet in our copy
                ocs_node_id = original_topology.get_ocs_for_logical_link(node1_id, node2_id, link_identifier)
                topo_view.logical_link_to_ocs_map[(node1_id, node2_id, link_identifier)] = ocs_node_id

                # Set according circuit
                ocs = topo_view.get_node(ocs_node_id)
                ocs.set_circuit(node1_id, node2_id)
                node1.connect(node2, link_identifier)
                node2.connect(node1, link_identifier)

        return TopologyWithOcsView(
            topology_copy=topo_view,
            relevant_link_identifier=topology_view_configuration.relevant_link_identifiers,
            relevant_ocs_identifier=topology_view_configuration.relevant_ocs_identifiers
        )


class OCSSwitchTypes(object):
    rotor = "RotorSwitch"
    ocs = "OpticalCircuitSwitch"
    list = [rotor, ocs]
