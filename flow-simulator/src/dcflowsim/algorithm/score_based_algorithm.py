import abc
import logging
import operator
from decimal import Decimal
from collections import defaultdict

from sortedcontainers import SortedList

from .abstract_algorithm import AbstractAlgorithmConfiguration, AbstractAlgorithmFactory
from .flow_score import SCORE_FACTORY
from .algorithm_factory import algorithm_factory_decorator

from dcflowsim import constants
from dcflowsim.environment import environment_listener
from dcflowsim.network import topology_factories, ocs_topology, topology, network_elements
from dcflowsim.utils.flow_handling import get_tor_node_ids_of_flow
from dcflowsim.network.networkx_wrapper import SetEdgesWithoutCapacityToInfinityWeight
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import AlgorithmInterface


class SortedScoreFlowList(environment_listener.EnvironmentListener):
    """
    Class uses a sorted to list to keep the order of flows ordered. Algorithm then should always try to allocate flow
    with highest scores_list (based on the scores_list implementation).
    """

    def __init__(self, env, score_factory=None):
        """
        Init the sorted scores_list flow list. Needs a factory.

        Args:
            score_factory: factory that produces the scores_list. Implement a scores_list if needed.
        """
        super(SortedScoreFlowList, self).__init__(
            priority=constants.ENVIRONMENT_VIEW_PRIORITY)
        self.__logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._environment = env

        self.__sorted_list = SortedList()
        self.__score_factory = score_factory

        self.__has_been_initialized = False

    @property
    def sorted_list(self):
        return self.__sorted_list

    def initialize(self, flows):
        """

        Args:
            flows: List of flows to be considered for initialization

        Returns:

        """
        if self.__has_been_initialized:
            # Do this only once
            return
        self.__has_been_initialized = True
        if flows is None:
            return
        for new_flow in flows:
            scored_flow = self._produce_score(new_flow)
            self.__sorted_list.add(scored_flow)
        # Finally resort flows
        self.resort()

    def get_sub_listeners(self):
        return list()

    def __len__(self):
        return len(self.__sorted_list)

    def __repr__(self):
        return self.__sorted_list.__repr__()

    def resort(self):
        """
        Resort method is need as there is not automated trigger if the objects of the list change their attributes
        that are used for sorting.

        """
        if self.__score_factory.dynamic():
            self.__sorted_list = SortedList(self.__sorted_list[:])

    def _produce_score(self, to_be_scored_flow):
        """
        Call the internal factory to produce a scores_list for this flow. This function actually creates a
        scores_list object, i.e., it calls the scores_list class.__init__ with the to_be_scored_flow as a parameter.

        Args:
            to_be_scored_flow: flow which should be scored.

        Returns:
            the scored flow object.

        """
        return self.__score_factory(to_be_scored_flow)

    def add_flow(self, new_flow):
        """
        Add a new FlowScore implementing AbstractFlowScore to the sorted list.

        Args:
            new_flow: to be added

        """
        scored_flow = self._produce_score(new_flow)
        self.__sorted_list.add(scored_flow)
        self.resort()
        self.__logger.info(f"Add flow {new_flow.global_id}")

    def remove_flow(self, old_flow):
        """
        Remove flow by its flow id.

        Args:
            old_flow: to be removed from the list

        """

        flow_id = old_flow.global_id

        idx_to_find = None
        for idx, value in enumerate(self.__sorted_list):
            if value.flow.global_id == flow_id:
                idx_to_find = idx
                break

        self.__sorted_list.pop(idx_to_find)
        self.__logger.info(f"Remove flow {flow_id}")

    def notify_add_flow(self, new_flow):
        """ Environment Listener implementation for FlowArrivalEvent"""
        self.add_flow(new_flow)

    def notify_rem_flow(self, removed_flow):
        """ Implementation of function called for FlowRemovalEvent"""
        self.remove_flow(removed_flow)


class TorByTorSortedScoreFlowList(environment_listener.EnvironmentListener):
    """
    Class uses a sorted lists to keep the order of flows ordered and in addition separates them by their tor pair.
     Algorithm then should always try to allocate flow with highest scores_list (based on the scores_list
     implementation).
    """

    def __init__(self, env, sorted_flow_lists):
        """
        Init the sorted scores_list flow list. Needs a factory.

        Args:
            sorted_flow_lists: Dict of SortedScoreFlowLists keyed by src dst tor pair (node ids)
        """
        super(TorByTorSortedScoreFlowList, self).__init__(
            priority=constants.ENVIRONMENT_VIEW_PRIORITY)
        self.__logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._environment = env

        self.__dict_of_sorted_lists = sorted_flow_lists

        self.__has_been_initialized = False

    @property
    def sorted_list(self):
        return self.__dict_of_sorted_lists

    def values(self):
        return self.__dict_of_sorted_lists.values()

    def keys(self):
        return self.__dict_of_sorted_lists.keys()

    def items(self):
        return self.__dict_of_sorted_lists.items()

    def __getitem__(self, item):
        return self.__dict_of_sorted_lists[item]

    def _get_flow_key(self, flow_obj):
        src_tor, dst_tor = get_tor_node_ids_of_flow(flow_obj)
        if src_tor < dst_tor:
            return src_tor, dst_tor
        else:
            return dst_tor, src_tor

    def initialize(self, flows):
        """

        Args:
            flows: List of flows to be considered for initialization

        Returns:

        """
        if self.__has_been_initialized:
            # Do this only once
            return
        self.__has_been_initialized = True
        if flows is None:
            return
        for new_flow in flows:
            self.__dict_of_sorted_lists[self._get_flow_key(new_flow)].add_flow(new_flow)
        # Finally resort flows
        self.resort()

    def get_sub_listeners(self):
        return list()

    def __len__(self):
        num_flows = 0
        for _, v in self.__dict_of_sorted_lists.items():
            num_flows += len(v)
        return num_flows

    def __repr__(self):
        return f"{self.__class__.__name__} - len={len(self.__dict_of_sorted_lists)}"

    def resort(self):
        """
        Resort method is need as there is not automated trigger if the objects of the list change their attributes
        that are used for sorting.

        """
        for k in self.__dict_of_sorted_lists.keys():
            self.__dict_of_sorted_lists[k].resort()

    def add_flow(self, new_flow):
        """
        Add a new FlowScore implementing AbstractFlowScore to the sorted list.

        Args:
            new_flow: to be added

        """
        self.__dict_of_sorted_lists[self._get_flow_key(new_flow)].add_flow(new_flow)
        self.__logger.info(f"Add flow {new_flow.global_id}")

    def remove_flow(self, old_flow):
        """
        Remove flow by its flow id.

        Args:
            old_flow: to be removed from the list

        """
        self.__dict_of_sorted_lists[self._get_flow_key(old_flow)].remove_flow(old_flow)
        self.__logger.info(f"Remove flow {old_flow.global_id}")

    def notify_add_flow(self, new_flow):
        """ Environment Listener implementation for FlowArrivalEvent"""
        self.add_flow(new_flow)

    def notify_rem_flow(self, removed_flow):
        """ Implementation of function called for FlowRemovalEvent"""
        self.remove_flow(removed_flow)


class FlowListFactory(object):
    """
    A factory class for flow lists ... we take the generation of flow lists out of the init methods of the
    score-basd algorithms
    """

    @classmethod
    def generate_flow_list(cls, flow_list, env, score_factory, add_env_listener=True):
        """
        Generates a flow list object and also registers this at the environment. The method also takes an existing
        flow_list and then only registers this one at the environment.

        Args:
            flow_list: if existing, just register
            env: the env where we register
            score_factory: the factory to produce the list - can be a string or an object
            add_env_listener

        Returns:

        """
        if flow_list is None:
            if isinstance(score_factory, str):
                score_factory = SCORE_FACTORY[score_factory]
            flow_list = SortedScoreFlowList(env, score_factory=score_factory)

        if add_env_listener:
            # Register the flow_list listener at the environment
            env.add_environment_listener(flow_list)

        return flow_list

    @classmethod
    def generate_tor_by_tor_flow_list(cls, flow_list, env, score_factory, tors, add_env_listener=True):
        import itertools as it
        src_dst_tor_to_flows = dict()
        for src_tor, dst_tor in it.combinations(sorted(tors), r=2):
            src_dst_tor_to_flows[(src_tor.node_id, dst_tor.node_id)] = FlowListFactory.generate_flow_list(
                flow_list=flow_list,
                env=env,
                score_factory=score_factory,
                add_env_listener=False
            )

        tor_by_tor_flow_list = TorByTorSortedScoreFlowList(env, src_dst_tor_to_flows)
        if add_env_listener:
            # Register the flow_list listener at the environment
            env.add_environment_listener(tor_by_tor_flow_list)

        return tor_by_tor_flow_list


class AbstractScoreBasedAlgorithm(environment_listener.EnvironmentListener):
    """
    Interface for algorithms using a scores_list to rate their flows.
    """

    def __init__(self, env, logger_name, flow_list, topology_view):
        super(AbstractScoreBasedAlgorithm, self).__init__(priority=constants.ALGORITHM_PRIORITY)

        self.__logger = logging.getLogger(logger_name)

        self._env = env

        assert flow_list is not None
        self._sorted_flow_list = flow_list

        assert topology_view is not None
        self._topology_view = topology_view
        self._network_graph = topology_view.network_graph

    @property
    def env(self):
        return self._env

    @property
    def topology_view(self):
        return self._topology_view

    @property
    def sorted_flow_list(self):
        return self._sorted_flow_list

    def initialize(self, flows):
        self._sorted_flow_list.initialize(flows)

    def get_sub_listeners(self):
        return [self.sorted_flow_list]

    def notify_rem_flow(self, removed_flow):
        """
        Removes flow from internal list and then tries to allocate flows with highest scores.

        Args:
            removed_flow: to be removed.

        """
        self.__logger.debug("Removed flow from topology copy.")
        self._allocate_flows()

    def notify_add_flow(self, new_flow):
        """
        Called when a new flow was added to the environment.

        Args:
            new_flow: to be added.

        """
        self._allocate_flows()

    @abc.abstractmethod
    def _allocate_flows(self):
        raise NotImplementedError


class GreedyOpticalScoreBasedAlgorithm(environment_listener.EnvironmentListener):
    """
    Algorithm for allocating demand-aware flows in Cerberus topology
    """
    class FlowAllocationError(Exception):
        pass

    def __init__(self, env, topology_view, src_dst_tor_to_flows, raise_no_allocation, allow_multi_allocation):
        super(GreedyOpticalScoreBasedAlgorithm, self).__init__(constants.ALGORITHM_PRIORITY)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._raise_no_allocation = raise_no_allocation
        self._allow_multi_allocation = allow_multi_allocation
        self._topology_view = topology_view
        self.environment = env
        self._ocss = self._topology_view.get_nodes_by_type(ocs_topology.OCSSwitchTypes.ocs)
        # Pending flows sorted by src dst ToR pair and ordered according to flow list - no circuit requested yet
        self._pending_flows = src_dst_tor_to_flows
        # Pre-allocated flow sorted by OCS and src-dst ToR pair(i.e., the circuit to set)- waiting for circuit to be set
        self._preallocated_flows = {ocs: dict() for ocs in self._ocss}
        # Allocated flows (with set circuit and rate > 0) sorted by OCS (circuit can be obtained via flow src and dst)
        self._allocated_flows_to_ocs = dict()

        # Keep track of free circuits/ocs per ToR
        self._available_ocs_per_tor = dict()  # TorOcsPair -> set

        ocs_ids = [ocs.node_id for ocs in self._ocss]
        for tor in self._topology_view.get_nodes_by_type(topology.KNOWN_SWITCH_TYPE_TOR_SWITCH):
            self._available_ocs_per_tor[tor.node_id] = set(ocs_ids)

        # Temp variable holding ids of newly allocated flows
        self.__newly_allocated_flows = list()

    @property
    def raise_no_allocation(self):
        return self._raise_no_allocation

    @property
    def num_preallocated_flows(self):
        return len(self.preallocated_flows_list)

    @property
    def preallocated_flows_list(self):
        prealloc_flows_list = list()
        for circuit_flows in self._preallocated_flows.values():
            prealloc_flows_list += list(circuit_flows.values())
        return prealloc_flows_list

    @property
    def allocated_flows_list(self):
        alloc_flows_list = list()
        for flows in self._allocated_flows_to_ocs.values():
            alloc_flows_list += flows
        return alloc_flows_list

    @property
    def not_allocated_flows(self):
        """
        Return list of flows that are known to this algorithm but not preallocated or allocated. To be used for flow
        re-distribution
        Returns:

        """
        prealloc_flows_list = self.preallocated_flows_list
        # Preallocated_flows holds SortedScoreLists so it's a bit messy here...
        return [f.flow for flows in self._pending_flows.values() for f in
                filter(lambda x: x.flow not in prealloc_flows_list,
                       flows.sorted_list)]  # bring a little pythonic brainfuck in...

    def remove_pending_flow(self, flow_obj):
        self._pending_flows[self._get_flow_key(flow_obj)].remove_flow(flow_obj)

    def get_sub_listeners(self):
        return []

    def initialize(self, flows):
        """
        Sort assigned flows by src and dst tor and order them by volume (desc). Then try to allocate as many as possible
        Returns:

        """
        for flow_obj in flows:
            self._pending_flows[self._get_flow_key(flow_obj)].add_flow(flow_obj)

        circuits_to_request = self._try_find_circuits_for_flows_all_tor_pairs()
        # Set circuits in environment
        for ocs, circuits in circuits_to_request.items():
            self.environment.set_optical_circuits(ocs.node_id, circuits)

    def notify_add_flow(self, new_flow):
        """
        Add flow to data structure and try to allocate. If not possible and raise_no_allocation=True raise Exception
        Args:
            new_flow:

        Returns:

        """
        try:
            ocs, src_tor, dst_tor = self._try_find_circuit_for_flow(new_flow)
            self._preallocated_flows[ocs][(src_tor, dst_tor)] = new_flow
            self.logger.debug(f"Allocate circuit between {src_tor} {dst_tor} on OCS {ocs.node_id} for {new_flow}")
            self.environment.set_optical_circuits(ocs.node_id, [get_tor_node_ids_of_flow(new_flow)])
        except GreedyOpticalScoreBasedAlgorithm.FlowAllocationError as e:
            self.logger.debug("Could not allocate flow")
            # Add flow to our list and try later again
            self._pending_flows[self._get_flow_key(new_flow)].add_flow(new_flow)
            if self._raise_no_allocation:
                raise e

    def notify_rem_flow(self, removed_flow):
        """
        Flow has finished. Release its circuit and try to allocate a pending flow. First try to re-use the circuit.
        If there is no flow extend search to all other ToR pairs
        Args:
            removed_flow:

        Returns:

        """
        ocs = self._allocated_flows_to_ocs[removed_flow.global_id]
        del self._allocated_flows_to_ocs[removed_flow.global_id]
        # Check if there is another flow for this circuit - we can directly allocate it
        for scored_flow in self._pending_flows[self._get_flow_key(removed_flow)].sorted_list:
            flow_obj = scored_flow.flow
            if not self._check_if_flow_is_routable(flow_obj, raise_exception=False):
                continue
            path = self._topology_view.get_shortest_path(flow_obj.source, flow_obj.destination, weight="weight")
            # Check if it is a single hop over the OCS only (path consists of nodes). Depending on where the flow starts
            # and ends, we have either:
            # Host -> Tor -> Tor -> Host = 4, Host -> Tor -> Tor = 3, Tor -> Tor -> Host = 3 or Tor -> Tor = 2
            if len(path) > 2 + isinstance(flow_obj.source, network_elements.Host) + \
                    isinstance(flow_obj.destination, network_elements.Host):
                # Only single hop paths so this path is too long. Check next flow
                continue
            rate = self._topology_view.network_graph.get_remaining_capacity_on_path(path)
            if rate > Decimal('0'):
                # Create a partial flow with no intermediate node and all routable volume
                partial_flow, subflow_id = flow_obj.create_subflow(
                    creation_time=self.environment.current_environment_time,
                    volume=flow_obj.routable_volume,
                    intermediate_nodes=[]
                )
                self.environment.allocate_flow(partial_flow, path, rate)

                # Change state of flow to allocated
                self._pending_flows[self._get_flow_key(removed_flow)].remove_flow(flow_obj)
                self._allocated_flows_to_ocs[flow_obj.global_id] = ocs
                # Allocated rate to another flow. return
                return [flow_obj.global_id]
        # Old circuit not useful now. Reset and check if we can set a new circuit
        rem_flow_src_tor_id, rem_flow_dst_tor_id = get_tor_node_ids_of_flow(removed_flow)
        # Add OCS back as available for ToRs
        self._available_ocs_per_tor[rem_flow_src_tor_id].add(ocs.node_id)
        self._available_ocs_per_tor[rem_flow_dst_tor_id].add(ocs.node_id)
        self.environment.release_optical_circuits(
            ocs.node_id, [(rem_flow_src_tor_id, rem_flow_dst_tor_id)]
        )
        # Try to allocate pending flows and get the circuits that have to be set. Flows that can be allocated on new
        # circuits will become pre-allocated flows
        self.__newly_allocated_flows = list()
        circuits_to_request = self._try_find_circuits_for_flows_all_tor_pairs()
        # Request the environment to apply the new circuits
        for ocs, circuits in circuits_to_request.items():
            self.environment.set_optical_circuits(ocs.node_id, circuits)
        return self.__newly_allocated_flows

    def notify_upd_ocs(self, ocs):
        """
        New circuits on OCS are active. If the ocs belongs to our topology-view, check for match in pre-allocated flow
         and allocate it to the new logical link. The pre-allocated flow will become an allocated flow
        Args:
            ocs: instance of OpticalCircuitSwitch which has new circuits set.

        Returns:

        """
        try:
            ocs = self._topology_view.get_node(ocs.node_id)
        except KeyError:
            # Not my OCS
            return
        # Get candidate flows that were pre-allocated to this OCS
        flows_to_assign_rate_to = self._preallocated_flows[ocs].values()
        for flow_obj in list(flows_to_assign_rate_to):
            src_tor_id, dst_tor_id = get_tor_node_ids_of_flow(flow_obj)
            other_circuit_endpoint = ocs.get_existing_circuit(src_tor_id)
            if other_circuit_endpoint is None or other_circuit_endpoint != dst_tor_id:
                continue

            # Get the shortest path for the flow. If we find one, the circuit for this flow has been set. If not
            # this notification belongs to circuit intended for another (pre-allocated) flow so continue our iteration
            path = self._topology_view.get_shortest_path(flow_obj.source, flow_obj.destination, weight="weight")
            if path is None or (len(path) > 2 + isinstance(flow_obj.source, network_elements.Host) + isinstance(
                    flow_obj.destination, network_elements.Host)):
                continue
                # Path should go only single-hop over the OCS (see comment in notify_remove_flow for details)
            rate = self._topology_view.network_graph.get_remaining_capacity_on_path(path)
            if rate == Decimal('0'):
                continue
            self.logger.debug(f"Allocate flow {flow_obj} with rate {rate} on {path}")
            # Create a partial flow with no intermediate node and all routable volume
            partial_flow, subflow_id = flow_obj.create_subflow(
                creation_time=self.environment.current_environment_time,
                volume=flow_obj.routable_volume,
                intermediate_nodes=[]
            )
            self.environment.allocate_flow(partial_flow, path, rate)

            # Change state of flow from pre-allocated to allocated
            del self._preallocated_flows[ocs][get_tor_node_ids_of_flow(flow_obj)]
            self._allocated_flows_to_ocs[flow_obj.global_id] = ocs

    def _get_flow_key(self, flow_obj):
        src_tor, dst_tor = get_tor_node_ids_of_flow(flow_obj)
        if src_tor < dst_tor:
            return src_tor, dst_tor
        else:
            return dst_tor, src_tor

    def _check_if_flow_is_routable(self, flow_obj, raise_exception=False):
        if (not self._allow_multi_allocation and flow_obj.current_rate > Decimal(
                '0')) or flow_obj.routable_volume <= Decimal('0'):
            # Flow already allocated somewhere else
            if raise_exception:
                raise GreedyOpticalScoreBasedAlgorithm.FlowAllocationError()
            return False
        return True

    def _try_find_circuit_for_flow(self, flow_obj):
        """
        Iterates over the OCSs and checks if we can set a circuit and allocate the flow to this new circuit.
        Throws exception if flow cannot be allocated.
        Args:
            flow_obj:

        Returns:
            ocs, src and dst ToR of the circuit
        Raises:
            GreedyOpticalScoreBasedAlgorithm.FlowAllocationError if we cannot allocate the flow
        """
        src_tor, dst_tor = get_tor_node_ids_of_flow(flow_obj)

        available_ocss = self._available_ocs_per_tor[src_tor].intersection(self._available_ocs_per_tor[dst_tor])
        if len(available_ocss) == 0:
            raise GreedyOpticalScoreBasedAlgorithm.FlowAllocationError()

        ocs_id_to_use = available_ocss.pop()
        self._available_ocs_per_tor[src_tor].remove(ocs_id_to_use)
        self._available_ocs_per_tor[dst_tor].remove(ocs_id_to_use)
        ocs = self._topology_view.get_node(ocs_id_to_use)
        return ocs, src_tor, dst_tor

    def _try_find_circuit_for_flows_of_tor_pair(self, flow_key):
        """
        Iterate over pending flows of ToR pair and try to allocate as many as possible. Returns the circuits that have
        to be set to allocate the flows. If flows can be allocated they are moved to the pre-allocated data-structure
        Args:
            flow_key: a.k.a tuple of src and dst tor node id

        Returns:
            dict keyed with OCS and values list of circuits to set for this OCS
        """
        circuits_per_ocs = defaultdict(list)
        # Do NOT iterate over lists that you modify...
        for scored_flow in list(self._pending_flows[flow_key].sorted_list):
            flow_obj = scored_flow.flow
            if not self._check_if_flow_is_routable(flow_obj, raise_exception=False):
                continue
            try:
                ocs, src_tor_id, dst_tor_id = self._try_find_circuit_for_flow(flow_obj)
                circuits_per_ocs[ocs].append((src_tor_id, dst_tor_id))
                self.logger.debug(
                    f"Allocate circuit between {src_tor_id} {dst_tor_id} on OCS {ocs.node_id} for {flow_obj}")
                # Change state of the flow to pre-allocated and remove from pending flows
                self._preallocated_flows[ocs][(src_tor_id, dst_tor_id)] = flow_obj
                self._pending_flows[self._get_flow_key(flow_obj)].remove_flow(flow_obj)
                self.__newly_allocated_flows.append(flow_obj.global_id)
            except GreedyOpticalScoreBasedAlgorithm.FlowAllocationError:
                # We don't care here that we cannot allocate a flow.
                pass
        return circuits_per_ocs

    def _try_find_circuits_for_flows_all_tor_pairs(self):
        """
        Iterate over all pairs of ToRs and try to allocate as many as possible of the pending flows. Returns the
        circuits that have to be set to allocate the flows.
        Returns:
            dict keyed with OCS and values list of circuits to set for this OCS
        """
        circuits_to_request = defaultdict(list)
        for flow_key in self._pending_flows.keys():
            this_circuits = self._try_find_circuit_for_flows_of_tor_pair(flow_key)
            for ocs, circuits in this_circuits.items():
                circuits_to_request[ocs] += circuits
        return circuits_to_request


@peewee_interface_factory_decorator(interface=AlgorithmInterface)
class GreedyOpticalScoreBasedAlgorithmConfiguration(AbstractAlgorithmConfiguration):
    def __init__(self, score_factory, raise_no_allocation=False, topology_view_configuration=None, flow_list=None,
                 allow_multi_allocation=True):
        super(GreedyOpticalScoreBasedAlgorithmConfiguration, self).__init__(
            factory=GreedyOpticalScoreBasedAlgorithmFactory.__name__
        )
        self.score_factory = score_factory
        self.topology_view_configuration = topology_view_configuration
        self.flow_list = flow_list
        self.raise_no_allocation = raise_no_allocation
        self.allow_multi_allocation = allow_multi_allocation

    @property
    def params(self):
        """
        Return params as dict.

        Returns: dict with attribute names and values

        """
        tmp = {
            "score_factory": self.score_factory,
            "raise_no_allocation": self.raise_no_allocation
        }
        if self.topology_view_configuration is not None:
            tmp["topology_view"] = self.topology_view_configuration.params
        if self.flow_list is not None:
            tmp["flow_list"] = self.flow_list
        if not self.allow_multi_allocation:
            tmp["allow_multi_allocation"] = self.allow_multi_allocation
        return tmp


@algorithm_factory_decorator
class GreedyOpticalScoreBasedAlgorithmFactory(AbstractAlgorithmFactory):
    @classmethod
    def generate_algorithm(cls, algorithm_configuration, env):
        import itertools as it
        topo_view = topology_factories.produce_view(algorithm_configuration.topology_view_configuration, env.topology)
        env.topology.add_view(topo_view)
        # Assumes that all ToRs are connected to all OCSs
        src_dst_tor_to_flows = FlowListFactory.generate_tor_by_tor_flow_list(
            flow_list=None,
            env=env,
            score_factory=SCORE_FACTORY[
                algorithm_configuration.score_factory
            ],
            tors=topo_view.get_nodes_by_type(topology.KNOWN_SWITCH_TYPE_TOR_SWITCH),
            add_env_listener=False
        )

        return GreedyOpticalScoreBasedAlgorithm(
            env=env,
            topology_view=topo_view,
            src_dst_tor_to_flows=src_dst_tor_to_flows,
            raise_no_allocation=algorithm_configuration.raise_no_allocation,
            allow_multi_allocation=algorithm_configuration.allow_multi_allocation
        )


class TorByTorGreedyScoreBasedAlgorithm(AbstractScoreBasedAlgorithm):
    """
    Class can implement greedy scores_list-based algorithm which tries to allocated flows on a Tor pair by ToR pair bases
    instead of all flows globally (intention is to speed up a bit). Flows per ToR pair are sorted based on their scores.
    """

    def __init__(self, env, topology_view=None, flow_list=None):
        """
        Init method.

        Args:
            env: the env as needed by algorithms.
            topology_view: Topology view to operate on
            flow_list: TorByTorSortedScoreFlowList for keeping track of the flows.
        """
        super(TorByTorGreedyScoreBasedAlgorithm, self).__init__(
            env,
            logger_name=__name__ + "." + self.__class__.__name__,
            topology_view=topology_view,
            flow_list=flow_list
        )
        self._env = env
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        # GreedyScoreBasedAlgorithm should use a specific flow_weighter, This will set it
        # inside of the NetworkXView
        self._network_graph.flow_weighter = SetEdgesWithoutCapacityToInfinityWeight()

    def _allocate_flows(self):
        """
        Method tries to allocate the flow with the highest scores_list.
        """
        edges_to_be_removed = []
        # Remove all edges that have currently 0 remaining capacity
        edges_to_be_removed += self._network_graph.remove_links(
            Decimal('0'),
            operator.le  # Lower equal because we want to get rid of zero capacity edges
        )
        for tor_pair, sorted_flows in self._sorted_flow_list.sorted_list.items():
            # (0) We need to resort the flows first because the ordering can change due to rate updates etc.
            sorted_flows.resort()
            # self.logger.info("The remaining flows: {}".format(len(sorted_flows.sorted_list)))

            # (1) Go through all scored flows of this ToR pair
            for scored_flow in sorted_flows.sorted_list:
                next_flow = scored_flow.flow

                # (2) try flow only if it has no rate (non preemptive, maybe lets do preemptive?!)
                if scored_flow.flow.current_rate > 0:
                    continue

                shortest_path = self._network_graph.get_shortest_path(
                    next_flow.source,
                    next_flow.destination,
                    weight="weight"
                )

                if shortest_path is not None:
                    # (3) Get remaining capacity of flow to the flow's rate
                    max_available_rate = self._network_graph.get_remaining_capacity_on_path(
                        shortest_path
                    )
                    if max_available_rate > 0:
                        self.logger.info(f"Max available rate: {max_available_rate}")
                        self._env.allocate_flow(
                            next_flow,
                            shortest_path,
                            max_available_rate
                        )
                        # We allocated a new flow. Update our temporary graph.
                        edges_to_be_removed += self._network_graph.remove_links(
                            Decimal('0'),
                            operator.le  # Lower equal because we want to get rid of zero capacity edges
                        )
                        continue
                # Exit loop and go to next ToR pair.
                break

        # Finally restore our old topology
        self._network_graph.add_links_from(edges_to_be_removed)


@peewee_interface_factory_decorator(interface=AlgorithmInterface)
class TorByTorGreedyScoreBasedAlgorithmConfiguration(AbstractAlgorithmConfiguration):
    def __init__(self, score_factory, topology_view_configuration=None, flow_list=None):
        super(TorByTorGreedyScoreBasedAlgorithmConfiguration, self).__init__(
            factory=TorByTorGreedyScoreBasedAlgorithmFactory.__name__
        )
        self.score_factory = score_factory
        self.topology_view_configuration = topology_view_configuration
        self.flow_list = flow_list

    @property
    def params(self):
        """
        Return params as dict.

        Returns: dict with attribute names and values

        """
        tmp = {"score_factory": self.score_factory}
        if self.topology_view_configuration is not None:
            tmp["topology_view"] = self.topology_view_configuration.params
        if self.flow_list is not None:
            tmp["flow_list"] = self.flow_list
        return tmp


@algorithm_factory_decorator
class TorByTorGreedyScoreBasedAlgorithmFactory(AbstractAlgorithmFactory):
    @classmethod
    def generate_algorithm(cls, algorithm_configuration, env):
        topo_view = topology_factories.produce_view(algorithm_configuration.topology_view_configuration, env.topology)
        env.topology.add_view(topo_view)
        flow_list = FlowListFactory.generate_tor_by_tor_flow_list(
            flow_list=None,
            env=env,
            score_factory=SCORE_FACTORY[
                algorithm_configuration.score_factory
            ],
            tors=topo_view.get_nodes_by_type(topology.KNOWN_SWITCH_TYPE_TOR_SWITCH)
        )

        return TorByTorGreedyScoreBasedAlgorithm(
            env,
            topo_view,
            flow_list
        )
