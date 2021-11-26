from abc import ABC, abstractmethod
from collections import UserDict
import itertools

import numpy as np

from dcflowsim import configuration
from dcflowsim.utils.number_conversion import *
from decimal import Decimal
from dcflowsim.utils.factory_decorator import factory_decorator
from .network_elements_factory import SWITCH_FACTORY


class Link(ABC):
    """
    Abstract link class.
    """
    def __init__(self, endpoint_1, endpoint_2, capacity, link_identifier=None, track_flows=True):
        assert endpoint_1 is not endpoint_2
        self.__endpoint_1 = endpoint_1
        self.__endpoint_2 = endpoint_2
        if link_identifier is None:
            link_identifier = LinkIdentifiers.default
        self.__link_identifier = link_identifier
        self.__total_capacity = convert_to_decimal(capacity)
        self.__free_capacity = convert_to_decimal(capacity)
        self.__flows = list()
        self.__track_flows = track_flows

    @property
    def free_capacity(self):
        return self.__free_capacity

    @property
    def total_capacity(self):
        return self.__total_capacity

    @property
    def link_identifier(self):
        return self.__link_identifier

    @property
    def flows(self):
        return self.__flows

    @property
    def id(self):
        return f"{self.__endpoint_1}-{self.__endpoint_2}" + \
               f"-{self.__link_identifier}" if self.__link_identifier is not None else ""

    def get_endpoints(self):
        return self.__endpoint_1, self.__endpoint_2

    def allocate(self, flow_obj):
        """
        Subtracts the flow's network route rate from the link capacity and adds the flow to the list of flows
        of the link
        Args:
            flow_obj: instance of flow

        Returns:

        """
        from dcflowsim.environment.flow import Flow, SuperFlow
        self.__free_capacity = self.__free_capacity - flow_obj.network_route.rate
        if self.__free_capacity < Decimal('0'):
            raise RuntimeError("Remaining capacity should not be negative: {}".format(self.__free_capacity))
        if not self.__track_flows:
            return
        if type(flow_obj) == Flow:
            self.__flows.append(flow_obj)
        elif type(flow_obj) == SuperFlow:
            self.__flows += [partial_flow for partial_flow in flow_obj.active_flows]

    def release(self, flow_obj):
        """
        Adds the flow's network route rate to the link capacity and removes the flow from the list of flows
        of the link
        Args:
            flow_obj: instance of flow

        Returns:

        """
        from dcflowsim.environment.flow import Flow, SuperFlow
        self.__free_capacity = self.__free_capacity + flow_obj.network_route.rate
        if self.__free_capacity > self.__total_capacity:
            raise RuntimeError("Remaining capacity should not be larger than total capacity")
        if not self.__track_flows:
            return
        if type(flow_obj) == Flow:
            self.__flows.remove(flow_obj)
        elif type(flow_obj) == SuperFlow:
            self.__flows -= [partial_flow for partial_flow in flow_obj.active_flows]

    def reset(self):
        """
        Resets the free capacity to total_capacity and deletes all references to allocated flows
        Returns:

        """
        self.__free_capacity = self.__total_capacity
        self.__flows = list()


class LinkIdentifiers(object):
    """ Container to store the string identifiers for different link types. Physical
    as well as logical link types."""
    default = 'default_link'  # Will be used if no identifier is given
    electric = 'electric_link'
    optical = 'optical_link'
    static = 'static_link'
    dynamic = 'dynamic_link'
    rotor = 'rotor_link'


class LinkContainer(UserDict):
    """
    A class implementing the userdict for storing link objects. As keys for link objects are tuples, and
    source or destination is invertable, keys in this class are interchangeable tuples.
    """

    def __missing__(self, key):
        """
        Implement __missing__.

        Args:
            key: a tuple

        Raises:
            KeyError if tuple is not in self.data.keys()
        """
        if not isinstance(key, tuple):
            raise KeyError(key)

        if (key[0], key[1]) not in self.data.keys() or (key[1], key[0]) not in self.data.keys():
            raise KeyError(key)

    def __contains__(self, node_tuple):
        """
        Implements __contains__.

        Args:
            node_tuple: (node_id_1, node_id_2)

        Returns:
            True if tuple is in keys.
        """
        return (node_tuple[0], node_tuple[1]) in self.data.keys() or (node_tuple[1], node_tuple[0]) in self.data.keys()

    def __setitem__(self, key, value):
        """
        Implements __setitem__.

        Args:
            key: tuple
            value: link object
        """
        if (key[1], key[0]) in self.data.keys():
            self.data[(key[1], key[0])] = value

        else:
            self.data[(key[0], key[1])] = value

    def __getitem__(self, item):
        """
        Implements __getitem__.

        Args:
            item: item key to look for.

        Returns:
            the value of the item.

        Raises:
            KeyError if item not found.
        """
        try:
            value = self.data[(item[0], item[1])]
            return value
        except KeyError:
            try:
                value = self.data[(item[1], item[0])]
                return value
            except KeyError:
                raise KeyError(item)

    def __delitem__(self, key):
        """
        Implements __delitem__.

        Args:
            key: item key to delete.

        Returns:


        Raises:
            KeyError if item not found.
        """
        try:
            del self.data[(key[0], key[1])]
        except KeyError:
            try:
                del self.data[(key[1], key[0])]
            except KeyError:
                raise KeyError(key)

    def get(self, key, default=None):
        """
        A custom get method. Should be the same as getitem.

        Args:
            key: to look up
            default: defualt value to return.

        Returns:
            value of key if found else None
        """
        try:
            return self.data[(key[0], key[1])]
        except KeyError:
            try:
                return self.data[(key[0], key[1])]
            except KeyError:
                return default


class MultiLinkMapper(UserDict):
    """ Implements a direction insensitive mapping between a (src, dest, ident) Link
    tuple and an object"""

    def __missing__(self, key):
        """ Implements __missing__.
        Args:
            key (tuple): Dictionary key, expects 3-tuple
        Raises:
            KeyError if key is no tuple or not in dictionary
        """
        if not isinstance(key, tuple):
            raise KeyError(key)

        if (key[0], key[1], key[2]) not in self.data.keys() or \
                (key[1], key[0], key[2]) not in self.data.keys():
            raise KeyError(key)

    def __contains__(self, key):
        """ Implements __contains__.
        Args:
            item (tuple): Dictionary key. 3-tuple
        Returns:
            True if tuple is in the dict keys; False else.
        """
        return (key[0], key[1], key[2]) in self.data.keys() or \
               (key[1], key[0], key[2]) in self.data.keys()

    def __setitem__(self, key, value):
        """Implements __setitem__.

        Args:
            key: Dictionary key, 3-tuple
            value: object to store
        """
        # Three tuple case
        if len(key) is 3:
            if (key[1], key[0], key[2]) in self.data.keys():
                self.data[(key[1], key[0], key[2])] = value
            else:
                self.data[key] = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        """
        Implements __getitem__

        Args:
            key (tuple): dictionary key, 3-tuple

        Returns:
            The value corresponding to the key

        Raises:
            KeyError is key is not stored in the dict
        """
        # Standard 3-tuple behavior
        # Additional for usage of 2-tuples/Single Link case/ None identifier
        if len(key) is 2:
            return self.__getitem__((key[0], key[1], LinkIdentifiers.default))

        elif len(key) is 3:
            if key[2] is None:
                return self.__getitem__((key[0], key[1], LinkIdentifiers.default))
            else:
                try:
                    value = self.data[key]
                    return value
                except KeyError:
                    value = self.data[(key[1], key[0], key[2])]
                    return value
        else:
            raise KeyError(key)

    def __delitem__(self, key):
        """
        Implements __delitem__
        Args:
            key (tuple): dictionary Key, 3-tuple/2-tuple

        Raises:
            KeyError if key is not in the dictionary
        """
        if len(key) is 3:
            try:
                del self.data[key]
            except KeyError:
                del self.data[(key[1], key[0], key[2])]
        elif len(key) is 2:
            self.__delitem__((key[0], key[1], LinkIdentifiers.default))

    def get(self, key, default=None):
        """
        Custom method with default argument to return if key is not in dict
        Args:
            key (tuple): dictionary key. 3-tuple
            default: Value to return if key not in dictionary

        Returns:
            The value for key, default if not found
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default


class MultiLinkContainer(MultiLinkMapper):
    """ Implements a userdictionary to store Link Object.
    Keys are (src, dest, ident) tuples. Source and destination are invertible.
    """

    def __setitem__(self, key, value):
        """Implements __setitem__. If the key is only a 2-tuple, the link identifier is
        inferred from the link/value object

        Args:
            key: Dictionary key, 3-tuple or 2-tuple
            value: Link object to store
        """
        # Three tuple case
        if len(key) is 3:
            # Call the base LinkMapper method
            super(MultiLinkContainer, self).__setitem__(key, value)
        elif len(key) is 2:
            self.__setitem__((key[0], key[1], value.link_identifier), value)
        else:
            raise KeyError(key)


class ElectricalLink(Link):
    """
    Class implementing an electrical link.
    """

    def __init__(self, node_1, node_2, capacity, link_identifier=None, track_flows=True):
        super(ElectricalLink, self).__init__(
            node_1,
            node_2,
            capacity=Decimal(capacity),
            link_identifier=link_identifier,
            track_flows=track_flows
        )


class OpticalLink(Link):
    pass


class Node(ABC):
    """
    Abstract Node class.
    """

    def __init__(self, node_id):
        self.__node_id = node_id
        self._links = dict()

    @property
    def links(self):
        return self._links

    @property
    def node_id(self):
        return self.__node_id

    @abstractmethod
    def connect(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self):
        raise NotImplementedError

    @abstractmethod
    def add_to_topology(self, topology):
        raise NotImplementedError

    def get_link(self, to_node, link_identifier=None):
        """
        Method to get a link to the to_node. If only node is g
        Args:
            to_node: node_id of the target node
            link_identifier: optional identifier of the type of link to use,

        Returns:
            A Link object

        Raises:
            KeyError if no link with given

        """
        if link_identifier is None:
            link_identifier = LinkIdentifiers.default

        return self._links[(to_node, link_identifier)]

    def get_links_to_node(self, to_node):
        """
        Method to get all links between self and the to node
        Args:
            to_node: node_id of the target node

        Returns:
            A list of link objects
        """
        return [link for key, link in self._links.items() if to_node in key]

    def add_link(self, link):
        # one endpoint of link should be this object
        end1, end2 = link.get_endpoints()

        if end1.node_id == self.node_id:
            self._links[(end2.node_id, link.link_identifier)] = link
            return True

        if end2.node_id == self.node_id:
            self._links[(end1.node_id, link.link_identifier)] = link
            return True

        raise RuntimeError("This node is not part of the link")

    def remove_link(self, link):
        # one endpoint of link should be this object
        end1, end2 = link.get_endpoints()

        if end1.node_id == self.node_id:
            del self._links[(end2.node_id, link.link_identifier)]
            return True

        if end2.node_id == self.node_id:
            del self._links[(end1.node_id, link.link_identifier)]
            return True

        raise RuntimeError("This node is not part of the link")

    @abstractmethod
    def copy(self):
        """
        Duplicates the object and the state if no external information or references are required
        Returns:

        """
        raise NotImplementedError

    def __lt__(self, other):
        return self.node_id < other.node_id


class AbstractSwitch(Node):
    """
    AbstractSwitch class. Implements already some functions.
    """

    def __init__(self, node_id, num_ports, network_identifier=None):
        super(AbstractSwitch, self).__init__(node_id)
        self.__network_identifier = network_identifier  # for instance: edge, core, or aggregation?!
        self.__num_ports = num_ports
        self.__logical_neighbors = dict()  # key: node_id, value: object
        self.__physical_neighbors = dict()

    @property
    def links(self):
        return self._links

    @property
    def logical_neighbors(self):
        """
        Neighbors in the sense that data can flow (L2)
        Returns:

        """
        return self.__logical_neighbors

    @property
    def physical_neighbors(self):
        """
        Other endpoints of links. Nodes that are directly connected via wire (L1)
        Returns:

        """
        return self.__physical_neighbors

    @property
    def num_ports(self):
        return self.__num_ports

    @property
    def network_identifier(self):
        return self.__network_identifier

    def add_link(self, link):
        if len(self.links.keys()) >= self.num_ports:
            raise RuntimeError("All ports are in use.")
        super(AbstractSwitch, self).add_link(link)

        end1, end2 = link.get_endpoints()
        if end1.node_id == self.node_id:
            self.physical_neighbors[end2.node_id] = end2
            return True

        if end2.node_id == self.node_id:
            self.physical_neighbors[end1.node_id] = end1
            return True
        raise RuntimeError("This node is not part of the link")

    def remove_link(self, link):
        super(AbstractSwitch, self).remove_link(link)
        end1, end2 = link.get_endpoints()
        if end1.node_id == self.node_id:
            del self.physical_neighbors[end2.node_id]
            return True

        if end2.node_id == self.node_id:
            del self.physical_neighbors[end1.node_id]
            return True
        raise RuntimeError("This node is not part of the link")


class AbstractSwitchConfiguration(configuration.AbstractConfiguration):
    """
    Abstract class for switch configuration. Use this to specify the electrical (packet) switches of a topology.
    The link_capacity can be used by topology factories to set the capacity of the electrical links.
    """

    def __init__(self, factory, link_capacity):
        super(AbstractSwitchConfiguration, self).__init__(
            factory=factory
        )
        self.__link_capacity = Decimal(link_capacity)

    @property
    def link_capacity(self):
        return self.__link_capacity

    @property
    def params(self):
        return self.__dict__


class ElectricalSwitch(AbstractSwitch):
    """
    Class implementing an electrical switch.
    """

    def __init__(self, node_id, num_ports, network_identifier=None):
        super(ElectricalSwitch, self).__init__(node_id, num_ports, network_identifier)

    def connect(self, other, link_identifier=None):
        """
        Connect, i.e., add a new logical neighbor. Link identifier helps to keep track of parallel links to the same
        logical neighbor
        Args:
            other:
            link_identifier:

        Returns:

        """
        if other.node_id in self.logical_neighbors:
            raise RuntimeError("Already connected")
        self.logical_neighbors[(other.node_id, link_identifier)] = other

    def disconnect(self, other, link_identifier=None):
        """
        Disconnect, i.e., remove a new logical neighbor. Link identifier helps to keep track of parallel links to the
        same logical neighbor
        Args:
            other:
            link_identifier:

        Returns:

        """
        if (other.node_id, link_identifier) not in self.logical_neighbors:
            raise RuntimeError("Not connected")
        del self.logical_neighbors[(other.node_id, link_identifier)]

    def get_neighbors(self):
        return sorted(list(set(n for n, _ in self.logical_neighbors)))

    def add_to_topology(self, topology):
        topology.add_switch(self)

    def copy(self):
        return ElectricalSwitch(
            self.node_id, self.num_ports, self.network_identifier
        )

    def __repr__(self):
        return f"ElectricalSwitch(node_id={self.node_id},num_ports={self.num_ports}," \
               f"network_identifier={self.network_identifier})"


class ElectricalSwitchFactory(object):
    def produce_switch(self, config, node_id, num_ports, network_identifier):
        return ElectricalSwitch(node_id, num_ports, network_identifier)


@factory_decorator(factory_dict=SWITCH_FACTORY, factory=ElectricalSwitchFactory)
class ElectricalSwitchConfiguration(AbstractSwitchConfiguration):
    """
    Configuration for basic electrical switch.
    """

    def __init__(self, link_capacity):
        super(ElectricalSwitchConfiguration, self).__init__(
            factory=ElectricalSwitchFactory.__name__,
            link_capacity=Decimal(link_capacity)
        )

    def __str__(self):
        return "{}-link_capacity={}".format(
            self.__class__.__name__,
            self.link_capacity
        )


class ElectricalSwitchWithOpticalPorts(AbstractSwitch):
    """
    Decorates an Electrical switch to add optical ports for connection to the OCS
    """

    def __init__(self, optical_ports, decorated_switch):
        super(ElectricalSwitchWithOpticalPorts, self).__init__(decorated_switch.node_id,
                                                               optical_ports,
                                                               decorated_switch.network_identifier)
        assert isinstance(decorated_switch, AbstractSwitch)
        self.__decorated_switch = decorated_switch

    @property
    def decorated_switch(self):
        return self.__decorated_switch

    def connect(self, other, link_identifier=None):
        self.decorated_switch.connect(other, link_identifier)

    def disconnect(self, other, link_identifier=None):
        self.decorated_switch.disconnect(other, link_identifier)

    def get_neighbors(self):
        return self.decorated_switch.get_neighbors()

    def add_to_topology(self, topology):
        topology.add_switch(self)

    @property
    def links(self):
        tmp = dict(self.decorated_switch.links)
        tmp.update(
            super(ElectricalSwitchWithOpticalPorts, self).links
        )
        return tmp

    @property
    def logical_neighbors(self):
        """
        Neighbors in the sense that data can flow (L2)
        Returns:

        """
        tmp = dict(self.decorated_switch.logical_neighbors)
        tmp.update(
            **super(ElectricalSwitchWithOpticalPorts, self).logical_neighbors
        )
        return tmp

    @property
    def physical_neighbors(self):
        """
        Other endpoints of links. Nodes that are directly connected via wire (L1)
        Returns:

        """
        tmp = dict(self.decorated_switch.physical_neighbors)
        tmp.update(
            **super(ElectricalSwitchWithOpticalPorts, self).physical_neighbors
        )
        return tmp

    @property
    def num_ports(self):
        return super(ElectricalSwitchWithOpticalPorts, self).num_ports + self.decorated_switch.num_ports

    @property
    def network_identifier(self):
        return super(ElectricalSwitchWithOpticalPorts, self).network_identifier

    def add_link(self, link):
        """

        Args:
            link:

        Returns:

        """
        if isinstance(link, OpticalLink):
            if len(self._links) >= super(ElectricalSwitchWithOpticalPorts, self).num_ports:
                raise RuntimeError("All ports are in use.")
            super(AbstractSwitch, self).add_link(link)
            end1, end2 = link.get_endpoints()
            if end1.node_id == self.node_id:
                super(ElectricalSwitchWithOpticalPorts, self).physical_neighbors[end2.node_id] = end2
                return True

            if end2.node_id == self.node_id:
                super(ElectricalSwitchWithOpticalPorts, self).physical_neighbors[end1.node_id] = end1
                return True
            raise RuntimeError("This node is not part of the link")
        else:
            self.decorated_switch.add_link(link)

    def remove_link(self, link):
        if isinstance(link, OpticalLink):
            super(AbstractSwitch, self).remove_link(link)
            end1, end2 = link.get_endpoints()
            if end1.node_id == self.node_id:
                del super(ElectricalSwitchWithOpticalPorts, self).physical_neighbors[end2.node_id]
                return True

            if end2.node_id == self.node_id:
                del super(ElectricalSwitchWithOpticalPorts, self).physical_neighbors[end1.node_id]
                return True
            raise RuntimeError("This node is not part of the link")
        else:
            self.decorated_switch.remove_link(link)

    def copy(self):
        return ElectricalSwitchWithOpticalPorts(
            optical_ports=self.num_ports, decorated_switch=self.decorated_switch.copy()
        )

    def __repr__(self):
        return f"ElectricalSwitchWithOpticalPorts(node_id={self.decorated_switch.node_id}," \
               f"num_el_ports={self.decorated_switch.num_ports},num_opt_ports={self.num_ports}," \
               f"network_identifier={self.decorated_switch.network_identifier})"


class ElectricalSwitchWithOpticalPortsFactory(object):
    def produce_switch(self, config, node_id, num_ports, network_identifier):
        return ElectricalSwitchWithOpticalPorts(
            optical_ports=config.num_optical_ports,
            decorated_switch=ElectricalSwitch(
                node_id,
                num_ports,
                network_identifier
            )
        )


@factory_decorator(factory_dict=SWITCH_FACTORY, factory=ElectricalSwitchWithOpticalPortsFactory)
class ElectricalSwitchWithOpticalPortsConfiguration(AbstractSwitchConfiguration):
    """
    Configuration for electrical switch with num_optical_ports ports that can connect to an OCS.
    """
    def __init__(self, num_optical_ports, link_capacity):
        super(ElectricalSwitchWithOpticalPortsConfiguration, self).__init__(
            factory=ElectricalSwitchWithOpticalPortsFactory.__name__,
            link_capacity=Decimal(link_capacity)
        )
        self.__num_optical_ports = Decimal(num_optical_ports)

    @property
    def num_optical_ports(self):
        return self.__num_optical_ports

    def __str__(self):
        return "{}-{}-link_capacity={}".format(
            self.__class__.__name__,
            self.__num_optical_ports,
            self.link_capacity
        )


class OpticalCircuitSwitchFactory(object):
    """ Factory for creating OCS """

    @classmethod
    def produce_switch(cls, config, node_id, num_ports, network_identifier):
        """ Returns an OCS, Config is ignored, since OCS odes not require additional
        arguments for creation. But SwitchFactory uses this signature"""
        return OpticalCircuitSwitch(node_id, num_ports, config.reconfig_delay, network_identifier)


@factory_decorator(factory_dict=SWITCH_FACTORY, factory=OpticalCircuitSwitchFactory)
class OpticalCircuitSwitchConfiguration(AbstractSwitchConfiguration):
    """ Config class for standard OCS switches"""

    def __init__(self, link_capacity, reconfig_delay=0):
        super(OpticalCircuitSwitchConfiguration, self).__init__(
            factory=OpticalCircuitSwitchFactory,
            # Not used when creating links in OCS topo, tolopogy config brings OCS link capacity
            link_capacity=convert_to_decimal(link_capacity)
        )
        self.__reconfig_delay = convert_to_decimal(reconfig_delay)

    @property
    def reconfig_delay(self):
        return self.__reconfig_delay

    def __str__(self):
        return "{}-link_capacity={}-reconfig_delay={}".format(
            self.__class__.__name__,
            self.link_capacity,
            self.__reconfig_delay
        )


class OpticalCircuitSwitch(AbstractSwitch):
    """
    Class implementing an optical switch.
    """

    def __init__(self, node_id, num_ports, reconfig_delay=0, network_identifier=None):
        super(OpticalCircuitSwitch, self).__init__(node_id, num_ports, network_identifier)
        self.__connection_matrix = np.zeros(shape=[num_ports, num_ports])
        # Available physical ports that are free
        self.__free_ports = set(range(num_ports))
        # Maps the id of a node to the physical port it is attached at the OCS
        self.__node_id_to_port_mapping = dict()
        self.__port_to_node_id_mapping = dict()
        self.night_duration = reconfig_delay

    @property
    def reconfig_delay(self):
        return self.night_duration

    @property
    def connection_matrix(self):
        return self.__connection_matrix

    def get_neighbors(self):
        return self.logical_neighbors

    def get_links(self):
        return self.links

    def add_link(self, link):

        # Here we update the physical links of the AbstractSwitch object
        super(OpticalCircuitSwitch, self).add_link(link)

        # Get other endpoint of link
        ep_1, ep_2 = link.get_endpoints()
        if ep_1.node_id == self.node_id:
            # ep_2 is the other one
            other_ep = ep_2
        elif ep_2.node_id == self.node_id:
            # ep_1 is the other one
            other_ep = ep_1
        else:
            raise RuntimeError("This node is not part of the link")

        # Get next free port number
        port = self.__free_ports.pop()
        self.__node_id_to_port_mapping[other_ep.node_id] = port
        self.__port_to_node_id_mapping[port] = other_ep.node_id

    def remove_link(self, link):
        super(OpticalCircuitSwitch, self).remove_link(link)
        # Get other endpoint of link
        ep_1, ep_2 = link.get_endpoints()
        if ep_1.node_id == self.node_id:
            # ep_2 is the other one
            other_ep = ep_2
        elif ep_2.node_id == self.node_id:
            # ep_1 is the other one
            other_ep = ep_1
        else:
            raise RuntimeError("This node is not part of the link")

        # Release port
        port = self.__node_id_to_port_mapping[other_ep.node_id]
        if port in self.__free_ports:
            raise RuntimeError("Port is already free. Something went terribly wrong")

        self.__free_ports.add(self.__node_id_to_port_mapping[other_ep.node_id])
        del self.__node_id_to_port_mapping[other_ep.node_id]
        del self.__port_to_node_id_mapping[port]

    def connect(self, other, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass

    def __get_port_of_endpoint(self, endpoint_id):
        try:
            return self.__node_id_to_port_mapping[endpoint_id]
        except KeyError:
            raise RuntimeError("Endpoint not attached to OCS.")

    def _get_endpoint_id_of_port(self, port):
        try:
            return self.__port_to_node_id_mapping[port]
        except KeyError:
            raise RuntimeError("Port not in use.")

    def set_circuit_by_port(self, port_1, port_2):
        if self.__connection_matrix[port_1, port_2]:
            # Ports are already connected.
            return
        if np.sum(self.__connection_matrix[:, port_1]) > 0 or \
                np.sum(self.__connection_matrix[:, port_2]) > 0:
            raise RuntimeError("Port already in use")
        self.__connection_matrix[port_1, port_2] = 1
        self.__connection_matrix[port_2, port_1] = 1
        assert np.array_equal(np.transpose(self.__connection_matrix), self.__connection_matrix)

    def set_circuit(self, endpoint_1_id, endpoint_2_id):
        """
        Sets circuit between two endpoints
        Args:
            endpoint_1_id (str): node id of first endpoint
            endpoint_2_id (str): node id of second endpoint
        """
        port_1 = self.__get_port_of_endpoint(endpoint_1_id)
        port_2 = self.__get_port_of_endpoint(endpoint_2_id)
        self.set_circuit_by_port(port_1, port_2)

    def release_circuit(self, endpoint_1_id, endpoint_2_id):
        """
        Releases circuit between two endpoints
        Args:
            endpoint_1_id (str): node id of first endpoint
            endpoint_2_id (str): node id of second endpoint
        """
        port_1 = self.__get_port_of_endpoint(endpoint_1_id)
        port_2 = self.__get_port_of_endpoint(endpoint_2_id)

        self.__connection_matrix[port_1, port_2] = 0
        self.__connection_matrix[port_2, port_1] = 0
        assert np.array_equal(np.transpose(self.__connection_matrix), self.__connection_matrix)

    def get_existing_circuit(self, endpoint_id):
        """
        Returns the id of the other endpoint of the circuit of the Node with endpoint_id.
        Returns None if no circuit is set.
        Args:
            endpoint_id:

        Returns:

        """
        port = self.__get_port_of_endpoint(endpoint_id)
        if np.sum(self.__connection_matrix[:, port]) == 0:
            # No circuit
            return None
        other_port = np.argmax(self.connection_matrix[port], axis=0)
        for candidate_ep_id, candidate_port in self.__node_id_to_port_mapping.items():
            if candidate_port == other_port:
                return candidate_ep_id
        raise RuntimeError("Inconsistent node id to port mapping.")

    def add_to_topology(self, topology):
        topology.add_optical_switch(self)

    def get_free_ports(self):
        """
        Method returns the free ports.

        Returns:
            List with node ids who are not yet connected via the optical switch.

        """
        # First get vector of free ports
        connection_vector = np.sum(self.connection_matrix, axis=0)

        # Create list that contains hosts which are physically connected but whose ports are not used (virtually)
        free_ports = []

        # Iterate through dict and check whether vector for this port is zero?!
        for node_id, port in self.__node_id_to_port_mapping.items():
            if connection_vector[port] == 0:
                free_ports.append(node_id)

        return free_ports

    def get_number_of_free_virtual_ports(self):
        """
        Return number of free virtual ports, i.e., ports that are physically connected but not used, i.e., where
            no circuit is set

        Returns:
            the number of free virtual ports.

        """
        connection_vector = np.sum(self.connection_matrix, axis=0)
        return self.num_ports - np.sum(connection_vector)

    def get_number_of_free_ports(self):
        """
        Return number of free (phyiscal) ports, i.e, where links could be added.

        Returns:
            the number of unused ports.

        """
        return len(self.__free_ports)

    def get_pairs_connected_nodes(self):
        """
        Returns a list of nodes that are currently connected as pairs of the node_ids
        Returns:
            list of pairs of node ids
        """
        pairs = list()
        for port1, port2 in itertools.combinations(range(self.num_ports), 2):
            if self.connection_matrix[port1, port2] == 0:
                continue
            node_1 = self._get_endpoint_id_of_port(port1)
            node_2 = self._get_endpoint_id_of_port(port2)
            if (node_2, node_1) in pairs:
                continue
            pairs.append((node_1, node_2))
        return pairs

    def release_all_circuits(self):
        self.__connection_matrix = np.zeros(shape=[self.num_ports, self.num_ports])

    def copy(self):
        return OpticalCircuitSwitch(
            node_id=self.node_id, num_ports=self.num_ports, reconfig_delay=self.reconfig_delay,
            network_identifier=self.network_identifier
        )

    def __repr__(self):
        return "{}(node_id={},num_ports={},network_identifier={})".format(
            self.__class__.__name__,
            self.node_id,
            self.num_ports,
            self.network_identifier
        )


class RotorSwitchFactory(object):
    """
    Factory class for RotorSwitches, used in OCS Topology to create RotorSwitches

    """

    @classmethod
    def produce_switch(cls, switch_config, node_id, num_ports, network_identifier):
        return RotorSwitch(matchings=switch_config.matchings, node_id=node_id, num_ports=num_ports,
                           day_duration=switch_config.day_duration, night_duration=switch_config.night_duration,
                           network_identifier=network_identifier)


@factory_decorator(factory_dict=SWITCH_FACTORY, factory=RotorSwitchFactory)
class RotorSwitchConfiguration(AbstractSwitchConfiguration):
    """
    Configuration class for RotorSwitches. RotorSwitchFactory creates a Switch from this.

    """

    def __init__(self, matchings, link_capacity, day_duration=constants.ROTORSWITCH_RECONF_PERIOD, night_duration=0):
        super(RotorSwitchConfiguration, self).__init__(
            factory=RotorSwitchFactory.__name__,
            link_capacity=convert_to_decimal(link_capacity)  # Link_capacity is currently not used in Topo factory
        )

        self.__matchings = matchings
        self.night_duration = night_duration
        self.day_duration = day_duration

    @property
    def matchings(self):
        return self.__matchings

    def __str__(self):
        return "{}-night={}-day={}-link_capacity={}".format(
            self.__class__.__name__,
            self.night_duration,
            self.day_duration,
            self.link_capacity
        )


class RotorSwitch(OpticalCircuitSwitch):
    """
    Inherits from OpticalCircuitSwitch to include rotor switch capabilities. Given a list of matchings, the RotorSwitch
    cycles through them.
    """

    def __init__(self, matchings, node_id, num_ports, day_duration=constants.ROTORSWITCH_RECONF_PERIOD,
                 night_duration=0, network_identifier=None):
        """ Inits the RotorSwitch

        reconfig_period and reconfig_offset are in unit of calls to the process() function
        of RotorReconfigEvent. So the real time reconfiguration period is equal to:
        reconfig_period times the reschedule period of the RotorReconfigEvent which
        defaults to 1 unit of simulation time.

        To ensure correct behavior of the RotorSwitch advance matching, we need a first
        RotorReconfigEvent at timestamp <reconfig_period>.

        Args:
            matchings: list of port matchings which the RotorSwitch will cycle through
            node_id (int): identification of the switch
            num_ports (int): number of ports of the rotor switch
            night_duration (int): indicates the reconfiguration time of the rotorswitch
            network_identifier (string): Identifier to distinguish specific switches in a topology
        """
        super(RotorSwitch, self).__init__(node_id, num_ports, night_duration, network_identifier)
        self.matchings = matchings
        self.current_matching_index = 0
        self.number_of_matchings = len(matchings)
        self.__night_duration = convert_to_decimal(night_duration)
        self.__day_duration = convert_to_decimal(day_duration)

    def disconnect_matching(self, topology):
        for port1, port2 in self.matchings[self.current_matching_index]:
            node1 = self._get_endpoint_id_of_port(port1)
            node2 = self._get_endpoint_id_of_port(port2)
            if topology is not None:
                topology.disconnect_node_pair_via_rotorswitch(self, node1, node2)
            self.release_circuit(node1, node2)

        self.current_matching_index = (self.current_matching_index + 1) % self.number_of_matchings

    def connect_matching(self, topology):
        """
        Applies and returns the next matching in the matchings list if the wait parameter is 0.

        Args:
            topology:

        Returns:
            List of node mappings: if a new matching is applied
            None: else

        Raises:
            RuntimeError: if the switch has no current matching set and the next matching is therefore undefined
        """

        if self.current_matching_index is None:
            raise RuntimeError("RotorSwitch has no current matching set. advance_matching() not possible.")

        for port1, port2 in self.matchings[self.current_matching_index]:
            self.set_circuit_by_port(port1, port2)
            if topology is not None:
                topology.connect_node_pair_via_rotorswitch(self, self._get_endpoint_id_of_port(port1),
                                                           self._get_endpoint_id_of_port(port2))

    @property
    def day_duration(self):
        return self.__day_duration

    def copy(self):
        mycopy = RotorSwitch(matchings=self.matchings, node_id=self.node_id, num_ports=self.num_ports,
                             day_duration=self.day_duration, night_duration=self.reconfig_delay,
                             network_identifier=self.network_identifier)
        mycopy.current_matching_index = self.current_matching_index
        return mycopy

    def reset(self):
        self.current_matching_index = 0


class Host(Node):
    """
    Class implementing a host.
    """

    def __init__(self, node_id):
        super(Host, self).__init__(node_id)
        self.__neighbors = dict()

    @property
    def physical_neighbors(self):
        return self.__neighbors

    def connect(self, other, link_identifier=None):
        self.__neighbors[other.node_id] = other

    def disconnect(self, other, link_identifier=None):
        if other.node_id not in self.__neighbors:
            raise RuntimeError("Not connected to this host.")
        del self.__neighbors[other.node_id]

    def get_neighbors(self):
        return self.physical_neighbors

    def add_to_topology(self, topology):
        topology.add_host(self)

    def add_link(self, link):
        super(Host, self).add_link(link)
        end1, end2 = link.get_endpoints()
        if end1.node_id == self.node_id:
            self.physical_neighbors[end2.node_id] = end2
            return True

        if end2.node_id == self.node_id:
            self.physical_neighbors[end1.node_id] = end1
            return True

    def copy(self):
        return Host(self.node_id)

    def __repr__(self):
        return f"{self.__class__.__name__}(node_id={self.node_id})"
