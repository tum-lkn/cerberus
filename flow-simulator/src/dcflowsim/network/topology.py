from abc import ABC, abstractmethod
import copy
import functools

from dcflowsim import configuration
from dcflowsim.network import network_elements
from dcflowsim.network.networkx_wrapper import \
    NetworkXTopologyWrapper, NetworkXMultiGraphTopologyWrapper
from dcflowsim.utils.factory_decorator import factory_decorator
from .topology_factories import TOPOLOGY_FACTORIES, TOPOLOGY_VIEW_FACTORIES

KNOWN_SWITCH_TYPE_EDGE = "Edge"
KNOWN_SWITCH_TYPE_TOR_SWITCH = "TorSwitch"
KNOWN_SWITCH_TYPES = [KNOWN_SWITCH_TYPE_EDGE, KNOWN_SWITCH_TYPE_TOR_SWITCH]


def update_views(func):
    """
    Decorator to automagically update topology views
    Args:
        func:

    Returns:

    """

    @functools.wraps(func)
    def func_with_update_views(*args, **kwargs):
        res = func(*args, **kwargs)
        for view in args[0].views:
            # Remove self from args since this is the original topo
            getattr(view, func.__name__)(*args[1:], **kwargs)
        return res

    return func_with_update_views


class AbstractTopology(ABC):

    @abstractmethod
    def network_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_node(self, node_id):
        raise NotImplementedError

    @abstractmethod
    @update_views
    def add_link(self, node_1, node_2, **kwargs):
        raise NotImplementedError

    @abstractmethod
    @update_views
    def remove_link(self, node_1, node_2):
        """
        Removes a Link from the topology
        Args:
            node_1: object of type AbstractNode
            node_2: object of type AbstractNode

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_link(self, node_1, node_2):
        """
        Returns the link object of the link between node_1 and node_2
        Args:
            node_1: either instance of Node or the node id (int)
            node_2: either instance of Node or the node id (int)

        Returns:
            Link object
        Raises:
            RuntimeError if there is no link between node_1 and node_2
        """
        raise NotImplementedError

    @abstractmethod
    def links(self):
        raise NotImplementedError

    @abstractmethod
    def get_network_graph(self):
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_nodes_by_type(self, node_type):
        raise NotImplementedError

    @abstractmethod
    def get_hosts(self):
        raise NotImplementedError

    @abstractmethod
    def get_links(self):
        raise NotImplementedError

    @abstractmethod
    def get_shortest_path(self, source, destination, weight):
        raise NotImplementedError

    @abstractmethod
    @update_views
    def allocate_route(self, flow):
        raise NotImplementedError

    @abstractmethod
    @update_views
    def release_route(self, flow):
        raise NotImplementedError

    @abstractmethod
    @update_views
    def reset(self):
        """
        Clears all resource allocations from topology
        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_corresponding_tor_switch(self, node):
        """
        Returns the ToR Switch (node) of the given node if applicable. Otherwise, an exception is raised
        Args:
            node:

        Returns:
            instance of node, the ToR Switch
        Raises:
            RuntimeError
        """
        raise NotImplementedError

    @abstractmethod
    def add_view(self, topology_view):
        """
        Register a view (listener) at this topology
        Args:
            topology_view:

        Returns:

        """
        raise NotImplementedError


class BaseDataCenterTopology(AbstractTopology):
    """
    Class representing a basic topology consisting of hosts and
    electrical switches.

    """

    def __init__(self, network_graph_wrapper=None, topology_type="BaseDataCenter", track_flows_on_links=True):
        """
        Init the topology class. External factory builds correct topology.

        Args:
            topology_type: the type of this topology
        """
        self.__views = list()
        self.__track_flows_on_links = track_flows_on_links
        self.__switches = dict()
        self.__hosts = dict()
        # Default to use standard wrapper
        if network_graph_wrapper is None:
            self.__network_graph = NetworkXTopologyWrapper()
        else:
            self.__network_graph = network_graph_wrapper

        if isinstance(self.__network_graph, NetworkXTopologyWrapper):
            self.__links = network_elements.LinkContainer()
        elif isinstance(self.__network_graph, NetworkXMultiGraphTopologyWrapper):
            self.__links = network_elements.MultiLinkContainer()
        else:
            raise RuntimeError("No supported TopologyWrapper given ... ")

        self.__topology_type = topology_type

    @property
    def views(self):
        return self.__views

    @property
    def links(self):
        return self.__links

    @property
    def topology_type(self):
        """
        topology_type property.

        Returns:
            self.__topology_type
        """
        return self.__topology_type

    @property
    def network_graph(self):
        """
        network_graph property.

        Returns:
            self.__network_graph
        """
        return self.__network_graph

    @network_graph.setter
    def network_graph(self, network_graph_wrapper):
        """
        Set network_graph property.

        Args:
            G: to be set.

        Returns:

        """
        self.__network_graph = network_graph_wrapper

    def get_network_graph(self):
        """
        Return the network graph representation of this data center topology (normally a networkx graph instance)

        Returns:
            self.network_graph is a logical graph representation of this topology
        """
        return self.network_graph

    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        """
        This method adds a link between two node objects.

        Node objects can be ElectricalSwitches or Hosts

        Args:
            node_1: first node object
            node_2: second node object
            link_identifier
            **kwargs: link attributes containing stuff like capacity
        """
        assert node_1.node_id in self.__hosts.keys() or node_1.node_id in self.__switches.keys()
        assert node_2.node_id in self.__hosts.keys() or node_2.node_id in self.__switches.keys()
        assert "capacity" in kwargs

        link = network_elements.ElectricalLink(
            node_1,
            node_2,
            link_identifier=link_identifier,
            capacity=kwargs["capacity"],
            track_flows=self.__track_flows_on_links
        )

        # Add link to nodes
        node_1.add_link(link)
        node_2.add_link(link)

        # Add link to links of this topology object
        self.links[(
            node_1.node_id,
            node_2.node_id,
            link.link_identifier
        )] = link

        # Add link to network graph object
        self.network_graph.add_link(
            node_1,
            node_2,
            link_identifier=link.link_identifier,
            **{
                "remaining_capacity": link.total_capacity,
                "capacity": link.total_capacity
            }
        )

    def remove_link(self, node_1, node_2, link_identifier=None):
        """
        Remove a link from this network. This removes the link from the link dict, the involved nodes,
        and the network graph representation.

        Args:
            node_1: endpoint of the link
            node_2: endpoint of the link
            link_identifier
        """
        link = self.links[(node_1.node_id, node_2.node_id, link_identifier)]
        node_1.remove_link(link)
        node_2.remove_link(link)
        self.network_graph.remove_link(node_1, node_2, link_identifier)

    def add_switch(self, switch):
        """
        Add a switch to this topology.

        Args:
            switch: to be added.
        """
        if switch.node_id in self.__switches.keys():
            raise RuntimeError("Switch id has already been added.")

        self.__switches[switch.node_id] = switch

    def get_link(self, node_1, node_2, link_identifier=None):
        """
        Returns the link object of the link between node_1 and node_2
        Forwards the call to network_graph
        Args:
            node_1: either instance of Node or the node id (int)
            node_2: either instance of Node or the node id (int)
            link_identifier

        Returns:
            Link object
        Raises:
            RuntimeError if there is no link between node_1 and node_2
        """
        # return self.network_graph.get_link(node_1, node_2)["link_object"]

        if isinstance(node_1, network_elements.Node) and isinstance(node_2, network_elements.Node):
            return self.links[(node_1.node_id, node_2.node_id, link_identifier)]
        else:
            return self.links[(node_1, node_2, link_identifier)]

    def get_switch(self, node_id):
        """
        Get switch by node_id

        Args:
            node_id: of the switch

        Returns:
            the switch

        Raises:
            KeyError if switch could not be found.
        """
        if node_id not in self.__switches.keys():
            raise KeyError("Switch not found!")

        return self.__switches[node_id]

    def get_host(self, node_id):
        """
        Get host by node_id

        Args:
            node_id: of the host

        Returns:
            the host

        Raises:
            KeyError if host could not be found.
        """
        if node_id not in self.__hosts.keys():
            raise KeyError("Host not found!")

        return self.__hosts[node_id]

    def add_host(self, host):
        """
        Add a host to this topology.

        Args:
            host: an instance of the Host class.

        Returns:
            True if operation was succesful

        """
        if host.node_id in self.__hosts.keys():
            raise RuntimeError("Host id has already been added: {} List of ids: {}".format(
                host.node_id,
                self.__hosts.keys()
            ))

        self.__hosts[host.node_id] = host

    def add_node(self, node, **kwargs):
        """
        Add an abstract network node.

        Function calls add_to_topology function of node object, which uses on
        Visitor pattern.

        Args:
            node: an abstract network node object.
            **kwargs: parameters of this network node.

        Returns:

        """
        node.add_to_topology(self)

    def get_node(self, node_id):
        try:
            return self.get_switch(node_id)
        except KeyError:
            try:
                return self.get_host(node_id)
            except KeyError:
                raise KeyError("Node not found")

    def get_nodes(self):
        """
        Get the network nodes objects: switches and hosts.

        Returns:
            list containing the switches and hosts.
        """
        return self.get_switches() + self.get_hosts()

    def get_nodes_by_type(self, node_type):
        if node_type not in KNOWN_SWITCH_TYPES:
            raise ValueError("Requested Node type not available in BaseDCTopo.")
        return [switch for switch in self.get_switches() if switch.network_identifier == node_type]

    def get_switches(self):
        """
        Get all switches.

        Returns:
            list with switches.
        """
        return list(self.__switches.values())

    def get_hosts(self):
        """
        Get all hosts.

        Returns:
            list with hosts
        """
        return list(self.__hosts.values())

    def get_links(self):
        """
        Returns list of links.

        Calls method of network_graph object.

        Returns:
            list of links.

        """
        return self.network_graph.get_links()

    def get_shortest_path(self, source, destination, weight):
        """
        Get the shortest path between source and destination.

        Function calls the get_shortest_path function of network_graph.

        Args:
            source: start node
            destination: end node
            weight: the weight to be used for shortest path calculation (e.g.,
            remaining capacity)

        Returns:
            a list of nodes belonging to shortest path.

        """
        return self.network_graph.get_shortest_path(
            source,
            destination,
            weight=weight
        )

    def allocate_route(self, flow):
        """
        Allocates resources for the provided flow's network route
        Args:
            flow: instance of flow

        Returns:

        """
        for link_tuple in flow.network_route.links:
            self.links[link_tuple].allocate(flow)
        self.network_graph.allocate_route(flow.network_route)

    def release_route(self, flow):
        """
        Releases resources for the provided flow's network route
        Args:
            flow: instance of flow

        Returns:

        """
        for link_tuple in flow.network_route.links:
            self.links[link_tuple].release(flow)
        self.network_graph.release_route(flow.network_route)

    def reset(self):
        for link_tuple in self.get_links():
            self.get_link(*link_tuple).reset()
            self.network_graph.reset_link(*link_tuple)

    def get_corresponding_tor_switch(self, node):
        """
        This topology has no notion of racks. Raise always RuntimeError
        Args:
            node:

        Returns:

        """
        raise RuntimeError("This topology has no notion of racks")

    def add_view(self, topology_view):
        if topology_view is self or topology_view in self.views:
            return
        self.views.append(topology_view)


class TopologyDecorator(AbstractTopology):
    """
    A topology decorator class. Takes an abstract topology and decorates it, i.e., the decorator adds additional
    functionality.
    """

    def __init__(self, decorated_topology, topology_type):
        self.__decorated_topology = decorated_topology
        self.__topology_type = topology_type

    @property
    def topology_type(self):
        return self.__topology_type

    @property
    def decorated_topology(self):
        return self.__decorated_topology

    @abstractmethod
    def add_node(self, node, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_link(self, node_1, node_2, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def remove_link(self, node_1, node_2):
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_links(self):
        raise NotImplementedError

    @abstractmethod
    def get_shortest_path(self, source, destination, weight):
        raise NotImplementedError

    def get_corresponding_tor_switch(self, node):
        """
        This topology has no notion of racks. Raise always RuntimeError
        Args:
            node:

        Returns:

        """
        raise RuntimeError("This topology has no notion of racks")


class AbstractTopologyConfiguration(configuration.AbstractConfiguration):
    def __init__(self, factory, decorated_topology=None):
        super(AbstractTopologyConfiguration, self).__init__(factory)
        self.decorated_topology_configuration = decorated_topology

    @property
    def params(self):
        raise NotImplementedError


class BaseDataCenterTopologyConfiguration(AbstractTopologyConfiguration):
    """ Configuration object for BaseDataCenter Topology. """

    def __init__(self, network_graph_wrapper=None):
        super(BaseDataCenterTopologyConfiguration, self).__init__(
            BaseDataCenterTopologyFactory.__name__
        )
        # Default to creating a SingleGraph
        if network_graph_wrapper is None:
            network_graph_wrapper = NetworkXTopologyWrapper()
        self.network_graph_wrapper = network_graph_wrapper

    @property
    def params(self):
        return {
            'network_graph_wrapper': self.network_graph_wrapper.__class.__name__
        }


class AbstractTopologyFactory(ABC):
    """
    A class implementing a topology factory.
    """

    @abstractmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        raise NotImplementedError


@factory_decorator(factory_dict=TOPOLOGY_FACTORIES)
class BaseDataCenterTopologyFactory(AbstractTopologyFactory):
    """ Factory for BaseDataCenterTopology """

    @classmethod
    def generate_topology(cls, topology_configuration, decorated_topology=None):
        """ Generates a BaseDataCenterTopology from the config object."""
        return BaseDataCenterTopology(network_graph_wrapper=
                                      topology_configuration.network_graph_wrapper)


class TopologyView(AbstractTopology):
    """
    Topology View for BaseDataCenter Topology. Provides filtered view on original topology. Modifications of the
    view happen locally, i.e., neither the original topology nor the other views are updated. Changes have to be applied
    to the original topology directly.
    """

    def __init__(self, topology_copy, relevant_link_identifier):
        assert isinstance(topology_copy, BaseDataCenterTopology)
        self.__topology_copy = topology_copy
        self.__relevant_link_identifier = relevant_link_identifier

    @property
    def network_graph(self):
        return self.__topology_copy.network_graph

    @property
    def links(self):
        return self.__topology_copy.links

    def add_node(self, node, **kwargs):
        node_copy = copy.deepcopy(node)
        self.__topology_copy.add_node(node_copy, **kwargs)

    def get_node(self, node_id):
        return self.__topology_copy.get_node(node_id)

    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        if link_identifier not in self.__relevant_link_identifier:
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

    def get_switch(self, node_id):
        return self.__topology_copy.get_switch(node_id)

    def get_host(self, node_id):
        return self.__topology_copy.get_host(node_id)

    def add_host(self, host):
        host = copy.deepcopy(host)
        self.__topology_copy.add_host(host)

    def get_switches(self):
        return self.__topology_copy.get_switches()

    def get_nodes(self):
        return self.__topology_copy.get_nodes()

    def get_nodes_by_type(self, node_type):
        return self.__topology_copy.get_nodes_by_type(node_type)

    def get_hosts(self):
        return self.__topology_copy.get_hosts()

    def get_links(self):
        return self.__topology_copy.get_links()

    def get_shortest_path(self, source, destination, weight):
        source = self.__topology_copy.get_node(source.node_id)
        destination = self.__topology_copy.get_node(destination.node_id)
        return self.__topology_copy.get_shortest_path(source, destination, weight)

    def allocate_route(self, flow):
        """
        Allocates the flow's network route on all links that are part of the view. Important for Host links!!
        Args:
            flow: instance of flow object

        Returns:

        """
        for link_tuple in flow.network_route.links:
            if link_tuple not in self.links:
                continue
            self.links[link_tuple].allocate(flow)
        self.network_graph.allocate_route(flow.network_route)

    def release_route(self, flow):
        """
        Releases the flow's network route on all links that are part of the view.
        Args:
            flow: instance of flow object

        Returns:

        """
        for link_tuple in flow.network_route.links:
            if link_tuple not in self.links:
                continue
            self.links[link_tuple].release(flow)
        self.network_graph.release_route(flow.network_route)

    def reset(self):
        for link_tuple in self.get_links():
            self.get_link(*link_tuple).reset()
            self.network_graph.reset_link(*link_tuple)

    def get_corresponding_tor_switch(self, node):
        node = self.__topology_copy.get_node(node.node_id)
        self.__topology_copy.get_corresponding_tor_switch(node)

    def add_view(self, topology_view):
        pass


class TopologyViewConfiguration(configuration.AbstractConfiguration):
    def __init__(self, relevant_link_identifiers):
        """

        Args:
            relevant_link_identifiers: list of link identifiers to use
        """
        super(TopologyViewConfiguration, self).__init__(TopologyViewFactory.__name__)
        self.relevant_link_identifiers = relevant_link_identifiers

    @property
    def params(self):
        return {
            'relevant_link_identifiers': self.relevant_link_identifiers
        }


@factory_decorator(factory_dict=TOPOLOGY_VIEW_FACTORIES)
class TopologyViewFactory(object):
    @classmethod
    def produce(cls, topology_view_configuration, original_topology):
        assert isinstance(original_topology, BaseDataCenterTopology)
        # First get deep copy of networkx wrapper
        topo_view = BaseDataCenterTopology(
            network_graph_wrapper=original_topology.network_graph.get_filtered_deep_copy(
                topology_view_configuration.relevant_link_identifiers
            ),
            track_flows_on_links=False
        )
        # Now add deepcopies of nodes
        for node_id, _ in topo_view.network_graph.get_nodes():
            # Get the node from the original topology
            node = original_topology.get_node(node_id)
            # Insert copy of node
            topo_view.add_node(node.copy())

        # Add link objects referencing to the our node copies
        for node_id_1, node_id_2, link_identifier in topo_view.network_graph.get_links():
            # Get the Link from the original topology
            try:
                link_object = original_topology.get_link(node_id_1, node_id_2, link_identifier)
                node1 = topo_view.get_node(node_id_1)
                node2 = topo_view.get_node(node_id_2)
                # Create new link object with same attributes but reference to copied nodes
                topo_view.add_link(node1, node2, link_identifier, capacity=link_object.total_capacity)
            except KeyError:
                # Link belongs to a decorating topology
                continue
        return TopologyView(
            topology_copy=topo_view,
            relevant_link_identifier=topology_view_configuration.relevant_link_identifiers
        )
