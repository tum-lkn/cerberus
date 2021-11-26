import copy
import logging
import itertools
import networkx as nx
import operator
import abc

from dcflowsim import literals
from . import network_elements
from dcflowsim.network.network_elements import LinkIdentifiers
from dcflowsim.utils.number_conversion import *

FLOW_WEIGHTERS = dict()


def flow_weighter_factory_decorator(cls):
    """
    Class decorator for all algorithm factories.

    Args:
        cls: the factory class that should be decorated.

    Returns:
        simply the cls
    """
    FLOW_WEIGHTERS[cls.__name__] = cls
    return cls


@flow_weighter_factory_decorator
class SetWeight(abc.ABC):
    @staticmethod
    def set_weight(attributes):
        raise NotImplementedError


@flow_weighter_factory_decorator
class SetRemainingCapacityAsWeight(SetWeight):
    @staticmethod
    def set_weight(attributes):
        #    for e, f, d in nx_graph.edges(data=True):
        attributes["weight"] = Decimal('1') / attributes["remaining_capacity"]


@flow_weighter_factory_decorator
class SetEdgesWithoutCapacityToInfinityWeight(SetWeight):
    @staticmethod
    def set_weight(attributes):
        remaining_capacity = attributes["remaining_capacity"]
        if remaining_capacity == Decimal('0.0'):
            attributes["weight"] = Decimal("Infinity")
        else:
            attributes["weight"] = Decimal('1.0')


def produce_flow_weighter(cls_name):
    return FLOW_WEIGHTERS[cls_name]()


class Path(object):
    """Class to encapsulate path with/without link identifiers for clean interface. """

    def __init__(self, node_list, link_identifiers=None):
        self.node_list = node_list
        self.link_identifiers = link_identifiers

    @property
    def links(self):
        """ Creates a list of links, either of form (src, dest) or (src, dest, ident).
        Depending if link identifiers are given."""
        if len(self.node_list) == 0:
            return []
        else:
            link_list = zip(self.node_list[:-1], self.node_list[1:])
            if self.link_identifiers is not None:
                link_list = [link + (ident, ) for link, ident in
                             zip(link_list, self.link_identifiers)]
            return list(link_list)

    def __len__(self):
        return len(self.node_list)


class NetworkXTopologyWrapper(object):
    """
    A topology wrapper for a networkx graph instance.
    """

    def __init__(self, G=None, flow_weighter=None):
        self.logger = logging.getLogger(
            __name__ + "." + self.__class__.__name__
        )
        self.__G = G
        if self.__G is None:
            self.__G = nx.Graph()
        self.__set_weight = False
        if flow_weighter is None:
            self._flow_weighter = SetRemainingCapacityAsWeight()
        else:
            self._flow_weighter = flow_weighter

    @property
    def G(self):
        return self.__G

    @G.setter
    def G(self, G):
        self.__G = G

    @property
    def flow_weighter(self):
        return self._flow_weighter

    @flow_weighter.setter
    def flow_weighter(self, flow_weighter):
        self._flow_weighter = flow_weighter

    def get_network_graph(self):
        return self.G

    def add_node(self, node, **kwargs):
        self.__G.add_node(node, **kwargs)

    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        if isinstance(node_1, network_elements.Node):
            node_id_1 = node_1.node_id
        else:
            node_id_1 = node_1

        if isinstance(node_2, network_elements.Node):
            node_id_2 = node_2.node_id
        else:
            node_id_2 = node_2
        self.__G.add_edge(node_id_1, node_id_2, **kwargs)

    def remove_link(self, node_1, node_2, link_identifier=None):
        self.__G.remove_edge(node_1.node_id, node_2.node_id)

    def update_link(self, node_id_1, node_id_2,
                    link_attribute, link_identifier=None):
        self.__G.edges[node_id_1, node_id_2].update(link_attribute)

    def set_flow_weights(self):
        for _, _, d in self.__G.edges(data=True):
            self.flow_weighter.set_weight(d)
        self.__set_weight = True

    def remove_nodes(self, node_container):
        """
        Function removes nodes from a network x topology

        Args:
            node_container: to be removed

        """
        self.__G.remove_nodes_from(node_container)

    def get_shortest_path(self, source, destination, weight=None):
        """
        Get the shortest path between source and destination node.

        Source and destination node can be either a node object or an id.

        Args:
            source: the source node or the source id
            destination: the destionation node or the destionation id
            weight: the weight for the shortest path calculation

        Returns:
            the shortest path list including source and destination id
        """

        if not self.__set_weight and weight is None:
            self.set_flow_weights()

        try:
            if isinstance(source, network_elements.Node):
                source = source.node_id
                destination = destination.node_id

            if weight is not None:
                node_list = nx.shortest_path(
                    self.__G,
                    source,
                    destination,
                    weight=weight
                )
            else:
                node_list = nx.shortest_path(
                    self.__G,
                    source,
                    destination
                )
            return Path(node_list)
        except nx.NetworkXNoPath:
            return None

    def remove_edges_without_capacity(self, key=literals.LINK_REMAINING_CAPACITY):
        # Look for edges with no capacity
        edges_to_remove = list()
        for e, f, d in self.__G.edges(data=True):
            if d[key] == 0:
                edges_to_remove.append((e, f))

        # Remove edges with no capacity
        for e, f in edges_to_remove:
            self.__G.remove_edge(e, f)

        return edges_to_remove

    def get_link(self, node_1, node_2, link_identifier=None):
        """
            Returns the link object of the link between node_1 and node_2
            Args:
                node_1: either instance of Node or the node id (int)
                node_2: either instance of Node or the node id (int)
                link_identifier:

            Returns:
                dict containing the edge data
            Raises:
                RuntimeError if there is no link between node_1 and node_2
        """
        if isinstance(node_1, network_elements.Node):
            node_1 = node_1.node_id
            node_2 = node_2.node_id
        return self.__G.edges[node_1, node_2]

    def get_remaining_capacity_on_path(self, path):
        max_available_rate = None
        for link_tuple in path.links:
            remaining_capacity = self.get_link(*link_tuple)["remaining_capacity"]
            max_available_rate = min(
                max_available_rate,
                remaining_capacity) if max_available_rate is not None else remaining_capacity
        return max_available_rate

    def remove_links(self, capacity_bound, relate=operator.lt):
        """
        A method that removes links with capacity less equal the lower capacity bound.

        Args:
            capacity_bound: links with less equal capacity will be removed.
            relate: should be an operator function

        """

        edges_to_be_removed = []
        for edge in self.G.edges(data=True):
            if isinstance(edge[0], str):
                if "OCS" in edge[0] or "OCS" in edge[1]:
                    break

            edge_remaining_capacity = edge[2][literals.LINK_REMAINING_CAPACITY]
            # if edge_remaining_capacity <= lower_capacity_bound:
            if relate(edge_remaining_capacity, capacity_bound):
                edges_to_be_removed.append(
                    edge
                )
        for e, f, _ in edges_to_be_removed:
            self.G.remove_edge(e, f)
        return edges_to_be_removed

    def add_links_from(self, links):
        """
        Encapsulate method to add links.

        Args:
            links: the links that should be add back to topology

        """
        self.logger.debug("Add links from: {}".format(links))

        self.G.add_edges_from(
            links
        )

    def allocate_route(self, route):
        for link_tuple in route.links:
            self.get_link(link_tuple[0], link_tuple[1])[literals.LINK_REMAINING_CAPACITY] \
                = self.get_link(link_tuple[0], link_tuple[1])[
                      literals.LINK_REMAINING_CAPACITY] - route.rate

            if self.get_link(link_tuple[0],
                             link_tuple[1])[literals.LINK_REMAINING_CAPACITY] < 0:
                raise RuntimeError("Remaining capacity should not be negative")

    def release_route(self, route):
        self.logger.debug("Release route: {}".format(route))
        for link_tuple in route.links:
            self.get_link(link_tuple[0], link_tuple[1])[literals.LINK_REMAINING_CAPACITY] \
                = self.get_link(link_tuple[0], link_tuple[1])[
                      literals.LINK_REMAINING_CAPACITY] + route.rate
            if self.get_link(link_tuple[0],
                             link_tuple[1])[literals.LINK_REMAINING_CAPACITY] > \
                    self.get_link(link_tuple[0],
                                  link_tuple[1])[literals.LINK_CAPACITY]:
                raise RuntimeError("Remaining capacity should not be larger than total capacity")

    def reset_link(self, node_1, node_2, link_identifier=None):
        if isinstance(node_1, network_elements.Node):
            node_1 = node_1.node_id
            node_2 = node_2.node_id
        link = self.get_link(node_1, node_2)
        link["remaining_capacity"] = link["capacity"]

    def reset(self):
        links_to_remove = list()
        for e, f in self.__G.edges():
            if "optical" in self.get_link(e, f):
                links_to_remove.append((e, f))
            else:
                self.reset_link(e, f)
        self.__G.remove_edges_from(links_to_remove)

    def get_links(self):
        return self.__G.edges

    def get_nodes(self):
        return self.__G.nodes(data=True)

    def get_deep_copy(self):
        graph_copy = self.__G.copy()
        for nid, data in self.__G.nodes(data=True):
            graph_copy.nodes[nid].update(copy.deepcopy(data))
        for e, f, data in self.__G.edges(data=True):
            graph_copy.edges()[(e, f)].update(copy.deepcopy(data))

        return NetworkXTopologyWrapper(
            G=graph_copy,
            flow_weighter=copy.deepcopy(self.flow_weighter)
        )

    def get_filtered_deep_copy(self, relevant_edge_identifiers):
        """
        Return a deep copy of the NetworkXTopology wrapper with edges filtered by identifier
        Returns:

        """
        # edge is a tuple (src, dest, ident)
        relevant_edges = [edge for edge in self.__G.edges
                          if edge[2].split("-")[0] in relevant_edge_identifiers]
        graph_copy = nx.edge_subgraph(self.__G, relevant_edges).copy()
        # Create also deepcopies of the data dicts
        for nid, data in graph_copy.nodes(data=True):
            graph_copy.nodes[nid].update(copy.deepcopy(data))
        for e, f, lid, data in graph_copy.edges(data=True, keys=True):
            graph_copy[(e, f, lid)].update(copy.deepcopy(data))

        return NetworkXTopologyWrapper(
            G=graph_copy,
            flow_weighter=copy.deepcopy(self.flow_weighter)
        )


class NetworkXMultiGraphTopologyWrapper(object):
    """ A Topology wrapper for a MultiGraph.
    Supports labeling edges as parts of 'Partial' Topologies"""

    def __init__(self, G=None, flow_weighter=None):
        # Get Logger
        self.logger = logging.getLogger(
            __name__ + "." + self.__class__.__name__
        )
        # Check if Graph is given
        self.__G = G
        if self.__G is None:
            self.__G = nx.MultiGraph()
        assert isinstance(self.__G, nx.MultiGraph), \
            "MultiGraphWrapper needs a 'nx.MultiGraph' as Graph"

        self.__set_weight = False
        if flow_weighter is None:
            self._flow_weighter = SetRemainingCapacityAsWeight()
        else:
            self._flow_weighter = flow_weighter

    @property
    def G(self):
        return self.__G

    @G.setter
    def G(self, G):
        self.__G = G

    @property
    def flow_weighter(self):
        return self._flow_weighter

    @flow_weighter.setter
    def flow_weighter(self, flow_weighter):
        self._flow_weighter = flow_weighter

    def get_network_graph(self):
        return self.G

    def add_node(self, node, **kwargs):
        self.__G.add_node(node, **kwargs)

    def add_link(self, node_1, node_2, link_identifier=None, **kwargs):
        """ Adds an edge to the graph and assigns the given tag to the edge"""
        if isinstance(node_1, network_elements.Node):
            node_id_1 = node_1.node_id
        else:
            node_id_1 = node_1

        if isinstance(node_2, network_elements.Node):
            node_id_2 = node_2.node_id
        else:
            node_id_2 = node_2

        if link_identifier is None:
            link_identifier = LinkIdentifiers.default

        self.__G.add_edge(node_id_1, node_id_2, key=link_identifier, **kwargs)

    def remove_link(self, node_1, node_2, link_identifier=None):
        if link_identifier is None:
            link_identifier = LinkIdentifiers.default
        self.__G.remove_edge(node_1.node_id, node_2.node_id, link_identifier)

    def update_link(self, node_id_1, node_id_2,
                    link_identifier=None, link_attribute=None):
        if link_identifier is None:
            link_identifier = LinkIdentifiers.default
        self.__G.edges[node_id_1, node_id_2, link_identifier].update(link_attribute)

    def set_flow_weights(self):
        for _, _, d in self.__G.edges(data=True):
            self.flow_weighter.set_weight(d)
        self.__set_weight = True

    def remove_nodes(self, node_container):
        """
        Function removes nodes from a network x topology

        Args:
            node_container: to be removed

        """
        self.__G.remove_nodes_from(node_container)

    def get_shortest_path(self, source, destination, weight=None):
        """
        Get the shortest path between source and destination node.

        Source and destination node can be either a node object or an id.

        Args:
            source: the source node or the source id
            destination: the destination node or the destination id
            weight: the weight for the shortest path calculation

        Returns:
            A Path
        """

        if not self.__set_weight and weight is None:
            self.set_flow_weights()

        try:
            if isinstance(source, network_elements.Node):
                source = source.node_id
                destination = destination.node_id

            if weight is not None:
                path = nx.shortest_path(
                    self.__G,
                    source,
                    destination,
                    weight=weight
                )
                identifiers = self.get_link_identifiers_on_shortest_path(path,
                                                                         weight=weight)
            else:
                path = nx.shortest_path(
                    self.__G,
                    source,
                    destination
                )
                identifiers = self.get_link_identifiers_on_shortest_path(path)

            return Path(path, identifiers)
        except nx.NetworkXNoPath:
            return None

    def remove_edges_without_capacity(self, key=literals.LINK_REMAINING_CAPACITY):
        # Look for edges with no capacity
        edges_to_remove = list()
        for e, f, identifier, d in self.__G.edges(data=True, keys=True):
            if d[key] == 0:
                #     edges_to_remove.append((e, f))
                # else:
                #     Use 1/remaining_capacity as weight -> favor links with high absolute amount of remaining capacity
                # d["weight"] = 1.0 / d[key]
                edges_to_remove.append((e, f, identifier))

        # Remove edges with no capacity
        for e, f, identifier in edges_to_remove:
            self.__G.remove_edge(e, f, identifier)

        return edges_to_remove

    def get_link(self, node_1, node_2, link_identifier=None):
        """
            Returns the link object of the link between node_1 and node_2
            Args:
                node_1: either instance of Node or the node id (int)
                node_2: either instance of Node or the node id (int)
                link_identifier:

            Returns:
                dict containing the edge data
            Raises:
                RuntimeError if there is no link between node_1 and node_2
        """
        if isinstance(node_1, network_elements.Node):
            node_1 = node_1.node_id
            node_2 = node_2.node_id

        if link_identifier is None:
            link_identifier = LinkIdentifiers.default

        return self.__G.edges[node_1, node_2, link_identifier]

    def get_link_identifiers_on_shortest_path(self, path, weight=None):
        """ Checks all the links on the given path. If multiple edges exist between a
        node pair, it finds the one with the minimum weight. If the weight is not present
        in the link_attributes, a weight of 1 is assumed for this link
        (this doubles the behavior of the nx.shortest_path method).
        If weight is None, the first found edge for a node pair is selected.

        Args:
            path: List of nodes to traverse
            weight: Link attribute to minimize in each step

        Returns:
            A list of link_identifiers that specify which link to take between each node pair
        """
        link_identifiers = []
        # Loop over node pairs in the path
        for e, f in zip(path[:-1], path[1:]):

            if weight is not None:
                weight_to_identifier_map = {}
                # get the link identifiers and the edge data for all edges between e and f
                # check for the smallest weight
                for identifier, data in self.__G.adj[e][f].items():
                    try:
                        weight_to_identifier_map[data[weight]] = identifier
                    except KeyError:
                        weight_to_identifier_map[1] = identifier
                min_weight = min(weight_to_identifier_map.keys())
                link_identifiers.append(weight_to_identifier_map[min_weight])
            else:
                # No weight given, take first edge that appears in graph
                link_identifiers.append(list(self.__G.get_edge_data(e, f).keys())[0])

        return link_identifiers

    def get_remaining_capacity_on_path(self, path):
        """ Calcs the remaining capactiy on the given path. In the MultiGraph case,
        the path is given by the nodes and the link_identifiers for the edges to use
        between two nodes.

        Args:
            path: List of nodes
        """
        max_available_rate = None

        tmp_links = path.links
        if path.link_identifiers is None:
            tmp_links = zip(tmp_links, [LinkIdentifiers.default for _ in range(len(path)-1)])

        for link_tuple in tmp_links:
            remaining_capacity = self.get_link(*link_tuple)["remaining_capacity"]
            max_available_rate = min(
                max_available_rate,
                remaining_capacity) if max_available_rate is not None else remaining_capacity
        return max_available_rate

    def remove_links(self, capacity_bound, relate=operator.lt):
        """
        A method that removes links with capacity less equal the lower capacity bound.

        Args:
            capacity_bound: links with less equal capacity will be removed.
            relate: should be an operator function

        """

        edges_to_be_removed = []
        for edge in self.G.edges(data=True, keys=True):

            # Hmm we need a global label and not a string here:
            if isinstance(edge[0], str):
                if "OCS" in edge[0] or "OCS" in edge[1]:
                    break

            edge_remaining_capacity = self.G.get_edge_data(
                edge[0],
                edge[1],
                edge[2]
            )[literals.LINK_REMAINING_CAPACITY]
            # if edge_remaining_capacity <= lower_capacity_bound:
            if relate(edge_remaining_capacity, capacity_bound):
                edges_to_be_removed.append(
                    edge
                )

        self.logger.debug("Remove capacity <= {} links {}".format(
            capacity_bound, edges_to_be_removed))

        for e, f, ident, _ in edges_to_be_removed:
            self.G.remove_edge(e, f, ident)
        return edges_to_be_removed

    def add_links_from(self, links):
        """
        Encapsulate method to add links.

        Args:
            links: the links that should be add back to topology

        """
        self.logger.debug("Add links from: {}".format(links))

        self.G.add_edges_from(
            links
        )

    def allocate_route(self, route):
        for link_tuple in route.links:
            self.get_link(link_tuple[0],
                          link_tuple[1],
                          link_tuple[2])[literals.LINK_REMAINING_CAPACITY] -= route.rate
            if self.get_link(link_tuple[0],
                             link_tuple[1],
                             link_tuple[2])[literals.LINK_REMAINING_CAPACITY] < 0:
                raise RuntimeError("Remaining capacity should not be negative")

    def release_route(self, route):
        self.logger.debug("Release route: {}".format(route))
        for link_tuple in route.links:
            self.get_link(link_tuple[0],
                          link_tuple[1],
                          link_tuple[2])[literals.LINK_REMAINING_CAPACITY] += route.rate
            if self.get_link(link_tuple[0],
                             link_tuple[1],
                             link_tuple[2])[literals.LINK_REMAINING_CAPACITY] > \
                    self.get_link(link_tuple[0],
                                  link_tuple[1],
                                  link_tuple[2])[literals.LINK_CAPACITY]:
                raise RuntimeError(
                    "Remaining capacity should not be larger than total capacity")

    def reset_link(self, node_1, node_2, link_identifier=None):
        if isinstance(node_1, network_elements.Node):
            node_1 = node_1.node_id
            node_2 = node_2.node_id

        if link_identifier is None:
            link_identifier = LinkIdentifiers.default

        link = self.get_link(node_1, node_2, link_identifier)
        link["remaining_capacity"] = link["capacity"]

    def reset(self):
        links_to_remove = list()
        for e, f, ident in self.__G.edges():
            if "optical" in self.get_link(e, f, ident):
                links_to_remove.append((e, f, ident))
            else:
                self.reset_link(e, f, ident)
        self.__G.remove_edges_from(links_to_remove)

    def get_links(self):
        # Return the edges as a list of tuples of form (src, dest, link_identifier)
        return self.__G.edges(keys=True)

    def get_nodes(self):
        return self.__G.nodes(data=True)

    def get_deep_copy(self):
        graph_copy = self.__G.copy()
        for nid, data in self.__G.nodes(data=True):
            graph_copy.nodes[nid].update(copy.deepcopy(data))
        for e, f, ident, data in self.__G.edges(data=True, keys=True):
            graph_copy.edges()[(e, f, ident)].update(copy.deepcopy(data))

        return NetworkXMultiGraphTopologyWrapper(
            G=graph_copy,
            flow_weighter=copy.deepcopy(self.flow_weighter)
        )

    def get_filtered_deep_copy(self, relevant_edge_identifiers):
        """
        Return a deep copy of the NetworkXTopology wrapper with edges filtered by identifier
        Returns:

        """
        # edge is a tuple (src, dest, ident)
        relevant_edges = [edge for edge in self.__G.edges if edge[2].split("-")[0] in relevant_edge_identifiers]
        graph_copy = nx.edge_subgraph(self.__G, relevant_edges).copy()
        # Create also deepcopies of the data dicts
        for nid, data in graph_copy.nodes(data=True):
            graph_copy.nodes[nid].update(copy.deepcopy(data))
        for e, f, lid, data in graph_copy.edges(data=True, keys=True):
            graph_copy.edges[(e, f, lid)].update(copy.deepcopy(data))

        return NetworkXMultiGraphTopologyWrapper(
            G=graph_copy,
            flow_weighter=copy.deepcopy(self.flow_weighter)
        )
