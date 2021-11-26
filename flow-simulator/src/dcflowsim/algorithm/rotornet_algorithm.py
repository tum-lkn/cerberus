"""
Implements the Rotornet Algorithm for the flow level simulator.
Based on http://cseweb.ucsd.edu/~gmporter/papers/sigcomm17-rotornet.pdf.
"""

__version__ = 0.1
__author__ = "Sebastian Schmidt, ga96vog@mytum.de, Kaan Aykurt, ge75kay@mytum.de, Johannes Zerwas, johannes.zerwas@tum.de"

import logging

from .abstract_algorithm import AbstractAlgorithmConfiguration, \
    AbstractAlgorithmFactory
from .algorithm_factory import algorithm_factory_decorator
from . import rotornet_strategy
from dcflowsim.constants import ALGORITHM_PRIORITY
from dcflowsim.environment import demand_matrix
from dcflowsim.environment.environment_listener import EnvironmentListener
from dcflowsim.network.network_elements import RotorSwitch
from dcflowsim.network.ocs_topology import OCSSwitchTypes
from dcflowsim.network.topology_factories import produce_view

from dcflowsim.utils.number_conversion import *
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import AlgorithmInterface

# lowest rate we will assign when creating indirect traffic to avoid rounding issues
ROTOR_NET_MIN_RATE = Decimal('10.0') ** - constants.FLOAT_DECIMALS


class RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocation(EnvironmentListener):
    """ Implements the RotorNet TwoHop Algorithm with full rate allocation.
    Needs to be used with a rotornet topology and SuperFlows as Flow class to support
    indirect TwoHop routing.
    """

    def __init__(self, environment, topology_view=None, tor2tor_demand_matrix=None, max_rate_per_flow=None,
                 allocate_indirect_flows_strategy=None):
        """ Constructor for TwoHop algorithm.

        Args:
            environment: The env used and listened to by the algorithm
        """
        EnvironmentListener.__init__(self, priority=ALGORITHM_PRIORITY)
        self.__environment = environment
        self.__tor_to_tor_demand_matrix = tor2tor_demand_matrix
        self._topology = topology_view
        self.rotor_switches = self.topology.get_nodes_by_type(OCSSwitchTypes.rotor)

        self.__MAX_RATE_PER_FLOW = max_rate_per_flow
        assert self.__MAX_RATE_PER_FLOW is not None

        self.__allocate_indirect_flows_strategy = allocate_indirect_flows_strategy
        assert self.__allocate_indirect_flows_strategy is not None
        self.__allocate_indirect_flows_strategy.set_rotornet_algorithm(self)

        # Stores the indirect traffic that needs to be prioritized
        self._indirect_subflows_per_rack_pair = dict()

        # Keep track of allocated flows for slot
        self._flow_dict = dict()
        # Keep track of remaining volume in slot between ToR pairs
        self._volume_dict = dict()
        # Keep track remaining volume that is allowed to be sent per flow in this slot
        self._remaining_volume_allowed_in_slot = dict()

        # Tracks the number of calls to notify_upd_ocs function
        self.__counter = 0

        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        self.__VOLUME_PER_SLOT = list(self.topology.ocs_links.values())[0].total_capacity

    @property
    def topology(self):
        return self._topology

    @property
    def max_rate_per_flow(self):
        return self.__MAX_RATE_PER_FLOW

    @property
    def volume_dict(self):
        return self._volume_dict

    @property
    def tor_to_tor_demand_matrix(self):
        return self.__tor_to_tor_demand_matrix

    def initialize(self, flows):
        self.__tor_to_tor_demand_matrix.initialize(flows)

    def get_sub_listeners(self):
        return [self.__tor_to_tor_demand_matrix]

    def notify_partial_rem_flow(self, removed_flow):
        """
        Function to process the removal of a partial flow
        Args:
            removed_flow: partial flow to be removed
        """

        for flow_list in self._flow_dict.values():
            # take only the list where the removed flow belongs to
            if removed_flow in flow_list:

                # remove the finished flow from that list
                flow_list.remove(removed_flow)

                # if there are still flows remaining
                if len(flow_list) > 0:
                    # take the next flow and route it
                    flow_to_allocate = flow_list[0]

                    path = self._topology.get_shortest_path(flow_to_allocate.source,
                                                            flow_to_allocate.destination,
                                                            weight=None)

                    assert len(path.links) <= 3

                    rate = self._topology.network_graph.get_remaining_capacity_on_path(path)
                    self.__environment.allocate_flow(flow_to_allocate, path, rate)

    def notify_upd_ocs(self, ocs):
        """ Function to process a reconfiguration of one ocs/the topology. Triggers a
        full rerouting, since all Flows are currently reset in the topology every time we
        reconfigure an OCS.

        Args:
            ocs: the ocs_that is changed
        """
        if type(ocs) is not RotorSwitch:
            return
        # Allocate flows once for each time update_rotornet_topology function is called in the environment
        nr_rotorswitches = len(self._topology.ocs)
        self.__counter += 1
        if self.__counter == nr_rotorswitches:
            self._allocate_flows()
            self.__counter = 0

    def _get_remaining_volume_of_flow_allowed_in_slot(self, flow_obj, volume_limit):
        """
        Returns the remaining volume that the given flow is allowed to send in the current slot.
        Args:
            flow_obj: flow to check
            volume_limit: allowed volume to be sent per flow in whole slot
                (is used if flow has not been allocated in this slot before)

        Returns:
            (Decimal) the volume that this flow can still send before hitting rate limit
        """
        if flow_obj.global_id not in self._remaining_volume_allowed_in_slot:
            # Flow has not been allocated yet for this slot
            self._remaining_volume_allowed_in_slot[flow_obj.global_id] = volume_limit
            return volume_limit
        else:
            rate_limited_volume_to_route = self._remaining_volume_allowed_in_slot[flow_obj.global_id]
            return rate_limited_volume_to_route

    def allocate_direct_traffic(self, tor_pair, volume_limit=None):
        """
        Allocates flows for direct transmission and updates the set of already allocated flows accordingly,
        decreases available volume per tor pair
        Args:
            tor_pair: tuple of ToR IDs of the src-dst pair to consider
            volume_limit (Decimal): Max volume per flow and slot

        Returns:
            allocated volume
        """
        total_allocated_volume = 0
        # Loop over all SuperFlows, spawn new direct SubFlows
        for super_flow in self.__tor_to_tor_demand_matrix.get_bidirectional_flows_by_tors(*tor_pair):
            if self._volume_dict[tor_pair] <= Decimal('0'):
                # No volume in slot left.
                return
            if super_flow.routable_volume <= Decimal('0'):
                # Flow has nothing to send, try next flow
                continue
            # Check if flow is already allocated (to enforce a rate limit per flow)
            rate_limited_volume_to_route = self._get_remaining_volume_of_flow_allowed_in_slot(super_flow, volume_limit)
            if rate_limited_volume_to_route <= Decimal('0'):
                # Flow is not allowed to send more, try next flow
                continue
            # Get min volume of remaining in slot, remaining routable of flow and allowed by rate limit
            rate_limited_volume_to_route = min(rate_limited_volume_to_route, super_flow.routable_volume,
                                               self._volume_dict[tor_pair])

            # Create partial flow with this volume and store it for later allocation
            partial_flow, subflow_id = super_flow.create_subflow(
                creation_time=self.__environment.current_environment_time,
                volume=rate_limited_volume_to_route
            )
            self._flow_dict[tor_pair] += [partial_flow]
            self._volume_dict[tor_pair] -= partial_flow.remaining_volume
            total_allocated_volume += partial_flow.remaining_volume

            ## Update already allocated volume
            # self._remaining_volume_allowed_in_slot[super_flow.global_id] -= rate_limited_volume_to_route
        return total_allocated_volume

    def allocate_direct_traffic_one_direction(self, tor_pair, volume_limit=None):
        """
        Allocates flows for direct transmission and updates the set of already allocated flows accordingly,
        decreases available volume per tor pair
        Args:
            tor_pair: tuple of ToR IDs of the src-dst pair to consider
            volume_limit (Decimal): Max volume per flow and slot

        Returns:
            allocated volume
        """
        total_allocated_volume = 0
        try:
            avail_volume = self._volume_dict[tor_pair]
            tor_key_pair = tor_pair
        except KeyError:
            avail_volume = self._volume_dict[(tor_pair[1], tor_pair[0])]
            tor_key_pair = (tor_pair[1], tor_pair[0])

        # Loop over all SuperFlows, spawn new direct SubFlows
        for super_flow in self.__tor_to_tor_demand_matrix.get_flows_by_tors(*tor_pair):
            if self._volume_dict[tor_key_pair] <= Decimal('0'):
                # No volume in slot left.
                return 0
            if super_flow.routable_volume <= Decimal('0'):
                # Flow has nothing to send, try next flow
                continue
            # Check if flow is already allocated (to enforce a rate limit per flow)
            rate_limited_volume_to_route = self._get_remaining_volume_of_flow_allowed_in_slot(super_flow, volume_limit)
            if rate_limited_volume_to_route <= Decimal('0'):
                # Flow is not allowed to send more, try next flow
                continue
            # Get min volume of remaining in slot, remaining routable of flow and allowed by rate limit
            rate_limited_volume_to_route = min(rate_limited_volume_to_route, super_flow.routable_volume,
                                               self._volume_dict[tor_key_pair])

            # Create partial flow with this volume and store it for later allocation
            partial_flow, subflow_id = super_flow.create_subflow(
                creation_time=self.__environment.current_environment_time,
                volume=rate_limited_volume_to_route
            )
            self._flow_dict[tor_key_pair] += [partial_flow]
            self._volume_dict[tor_key_pair] -= partial_flow.remaining_volume
            total_allocated_volume += partial_flow.remaining_volume

            ## Update already allocated volume
            # self._remaining_volume_allowed_in_slot[super_flow.global_id] -= rate_limited_volume_to_route
        return total_allocated_volume

    def initialize_flow_volume_dict_for_tor_pair_route_stored_indirect_traffic(self, rotor_switch, torA, torB):
        """
        Resets and initializes data structures for allocating flows in new slot and resets remaining volumes of that
        slot
        Args:
            rotor_switch (RotorSwitch): rotor switch that is currently considered.
            torA (str): id of first ToR
            torB (str): id of second ToR

        Returns:

        """
        # initialize flow and volume dictionaries
        assert (torA, torB) not in self._flow_dict or len(self._flow_dict[(torA, torB)]) == 0
        self._flow_dict[(torA, torB)] = []
        self._volume_dict[(torA, torB)] = \
            min(
                self._topology.ocs_links[
                    (torA, rotor_switch.node_id)].total_capacity,
                self._topology.ocs_links[
                    (torB, rotor_switch.node_id)].total_capacity
            ) * rotor_switch.day_duration

        # add the stored indirect flows to the dictionary and subtract their volumes from available volume
        # STORED INDIRECT TRAFFIC A --> B
        self._flow_dict[(torA, torB)] += \
            self._get_stored_indirect_flows(src_tor=torA, dest_tor=torB)

        # STORED INDIRECT TRAFFIC B --> A
        self._flow_dict[(torA, torB)] += \
            self._get_stored_indirect_flows(src_tor=torB, dest_tor=torA)

        for flow_obj in self._flow_dict[(torA, torB)]:
            self._volume_dict[(torA, torB)] -= flow_obj.remaining_volume

    def _allocate_flows(self):
        """ Checks all currently connected rack pairs. Then
        1: Routes indirect traffic that is stored
        2: Routes direct traffic
        3: Negotiates new indirect traffic
        between the connected paris.
        Current Flow order depends on the list ordering inside the
        Tor2Tor demand matrix. So no fair share but first come first serve.
        Indirect volumes are assigned greedy.
        """
        if not self.__tor_to_tor_demand_matrix.has_flows:
            # Demand matrix empty. Return
            self.logger.debug("Demand matrix empty. Return.")
            return
        self.logger.debug("Demand matrix full. Run allocation.")
        # Reset remaining volumes per flow
        self._remaining_volume_allowed_in_slot = dict()
        self._volume_dict = dict()

        #### NEW INDIRECT TRAFFIC ####
        self.__allocate_indirect_flows_strategy.allocate()

        allocated_flow = False
        # Allocate first flow in list for each matching
        for flow_list in self._flow_dict.values():
            # if there is a flow in the flow list
            if len(flow_list) > 0:
                # take the first one and route it
                flow_to_allocate = flow_list[0]

                path = self._topology.get_shortest_path(flow_to_allocate.source,
                                                        flow_to_allocate.destination,
                                                        weight=None)
                assert len(path.links) <= 3

                rate = self._topology.network_graph.get_remaining_capacity_on_path(path)

                self.__environment.allocate_flow(flow_to_allocate, path, rate)
                allocated_flow = allocated_flow or rate > 0

    def get_stored_indirect_volume_per_rack_pair(self, intermediate_tor, dest_tor):
        """ Calculates the sum of the indirect traffic stored on torA for torB

        Args:
            intermediate_tor (str): Node ID of the start ToR switch
            dest_tor (str): Node ID of the destination ToR switch
        Returns:
            Sum of volumes of indirect subflows from A --> B
        """
        try:
            volume = 0
            # Loop all subflows, get remaining volume of the active partial flows
            for subflow_id in \
                    self._indirect_subflows_per_rack_pair[(intermediate_tor, dest_tor)]:
                volume += self._get_partial_flow_from_subflow_id(subflow_id).remaining_volume
            return volume
        except KeyError:
            return 0

    def _get_partial_flow_from_subflow_id(self, subflow_id):
        """ Retrieves the corresponding SuperFlow object from the environment and uses it
        to return the partial flow object that is currently active on the given subflow

        Args:
            subflow_id (str): String identifier of the subflow
        Returns:
            Flow object that is the active partial flow for the subflow_id
        """
        # Get the SuperFlow ID from subflow ID: <SuperFlowID>_<SubFlowID>
        superflow_id = int(subflow_id.split("_")[0])

        # Get SuperFlow from environment and PartialFlow from SuperFlow and return
        return self.__environment.get_flow_by_id(superflow_id) \
            .get_active_partial_flow_from_subflow_id(subflow_id)

    def _add_subflow_to_intermediate_node(self, intermediate_tor, dest_tor, subflow_id):
        """ Adds the subflow_id to the subflows that are stored on intermediate_tor to go
        to dest_tor.

        Args:
            intermediate_tor (str): String ID of the storing ToR (intermediate node)
            dest_tor (str): String ID of the target ToR
        """
        try:
            self._indirect_subflows_per_rack_pair[(intermediate_tor, dest_tor)]. \
                append(subflow_id)
        except KeyError:
            self._indirect_subflows_per_rack_pair[(intermediate_tor, dest_tor)] \
                = [subflow_id]

    def allocate_indirect_flows_one_direction(self, src_tor_id, intermediate_tor_id, dst_tor_id, tor_key_pair,
                                              rotor_switch, volume_limit):
        """
        Tries to allcoate indirect traffic in one direction of the ToR pair
        Args:
            src_tor_id (str):
            intermediate_tor_id (str):
            dst_tor_id (str):
            tor_key_pair (tuple): basically either (src_tor_id, intermediate_tor_id) or
                (intermediate_tor_id, src_tor_id). The ToR pair on which the flows are to be allocated
            rotor_switch (RotorSwitch):
            volume_limit (Decimal): max. volume per slot and flow

        Returns:
            Pre-allocated volume
        """
        # Get the available volume
        try:
            avail_volume = self._volume_dict[tor_key_pair]
        except KeyError:
            avail_volume = self._volume_dict[(tor_key_pair[1], tor_key_pair[0])]
            tor_key_pair = (tor_key_pair[1], tor_key_pair[0])
        available_volume_a_b_c = self.get_available_indirect_volume(
            src_tor=src_tor_id,
            intermediate_tor=intermediate_tor_id,
            dest_tor=dst_tor_id,
            rotor_switch=rotor_switch,
            day_duration=rotor_switch.day_duration,
            avail_volume=avail_volume)

        # Get the Flows from A to C and create subflows via B for them
        flows, routed_volume = self._create_indirect_flows(
            src_tor=src_tor_id,
            intermediate_tor=intermediate_tor_id,
            dest_tor=dst_tor_id,
            available_volume_src_to_dest=available_volume_a_b_c,
            volume_limit=volume_limit
        )

        self._flow_dict[tor_key_pair] += flows
        self._volume_dict[tor_key_pair] -= routed_volume
        return routed_volume

    def _create_indirect_flows(self, src_tor, intermediate_tor, dest_tor, available_volume_src_to_dest,
                               volume_limit=None):
        """ Iterates over all flows from src to dest, routes them via intermediate_tor as
        long as volume is available

        Args:
            src_tor (str): NodeID of the source ToR
            intermediate_tor (str): NodeID of the intermediate ToR
            dest_tor (str): NodeID of the destination ToR
            available_volume_src_to_dest (Decimal): Available volume for indirect routing
            volume_limit (Decimal):
        """
        flows = list()
        total_routed = 0

        super_flows = \
            self.__tor_to_tor_demand_matrix.get_flows_by_tors(src_tor, dest_tor)

        for super_flow in super_flows:
            # Check if the link has volume left for the slot
            if round_decimal_down(available_volume_src_to_dest) <= Decimal('0'):
                # Return, no need to iterate further...
                return flows, total_routed
            # Check if super_flow has volume left, that is not bound to a subflow yet
            if super_flow.routable_volume <= Decimal('0'):
                continue
            rate_limited_volume_to_route = self._get_remaining_volume_of_flow_allowed_in_slot(
                super_flow, volume_limit
            )
            if rate_limited_volume_to_route <= Decimal('0'):
                # Flow is not allowed to send more, try next flow
                continue

            # Get the exact volume that we are able to transmit
            exact_subflow_volume = min(
                super_flow.routable_volume,
                rate_limited_volume_to_route,
                round_decimal_down(available_volume_src_to_dest)
            )

            # Create a flow with the real volume
            partial_flow, subflow_id = super_flow.create_subflow(
                creation_time=self.__environment.current_environment_time,
                volume=exact_subflow_volume,
                intermediate_nodes=[self._topology.get_node(intermediate_tor)])

            assert partial_flow.destination is \
                   self._topology.get_node(intermediate_tor)

            flows.append(partial_flow)

            # Reduce the routable volume between src and dest by the
            # assumed_volume to keep the algorithm from assigning rates that are
            # not available during the whole slot
            available_volume_src_to_dest -= exact_subflow_volume
            total_routed += exact_subflow_volume

            # Update already allocated volume
            self._remaining_volume_allowed_in_slot[super_flow.global_id] -= exact_subflow_volume

            # Store the subflow ID to prioritize it when intermediate_tor and
            # dest_tor get connected in the future.
            self._add_subflow_to_intermediate_node(intermediate_tor=intermediate_tor,
                                                   dest_tor=dest_tor,
                                                   subflow_id=subflow_id)

        return flows, total_routed

    def get_available_indirect_volume(self, src_tor, intermediate_tor, dest_tor,
                                      rotor_switch, day_duration, avail_volume):
        """ Checks for the available volume on the complete indirect path. Assumes that
        the RotorSwitches on the path have the same and a synchronized reconfiguration
        period and link capacities.

        Args:
            src_tor (str): NodeID of the source ToR
            intermediate_tor (str): NodeID of the intermediate ToR
            dest_tor (str): NodeID of the destination ToR
            rotor_switch: A RotorSwitch; Link capacities over this will be used
            day_duration: Time that the RotorSwitch holds a matching
            avail_volume:
        Returns:
            Available volume on Src --> Int --> Dest;
        """

        # Get total volume and subtract direct + indirect demand (bidirectionally), this
        # should be independent of whether the direct traffic was routed
        if avail_volume == Decimal('0'):
            return Decimal('0')
        total_volume_inter_to_dest = min(
            self._topology.ocs_links[
                (intermediate_tor, rotor_switch.node_id)].total_capacity,
            self._topology.ocs_links[
                (src_tor, rotor_switch.node_id)].total_capacity
        ) * day_duration

        direct_demand_inter_to_dest = \
            self.__tor_to_tor_demand_matrix.get_demand_by_node_ids(intermediate_tor, dest_tor) + \
            self.__tor_to_tor_demand_matrix.get_demand_by_node_ids(dest_tor, intermediate_tor)

        indirect_demand_inter_to_dest = \
            self.get_stored_indirect_volume_per_rack_pair(intermediate_tor, dest_tor) + \
            self.get_stored_indirect_volume_per_rack_pair(dest_tor, intermediate_tor)

        # Get the volume that is left if the direct demand is satisfied
        available_volume_inter_to_dest = max((total_volume_inter_to_dest
                                              - direct_demand_inter_to_dest
                                              - indirect_demand_inter_to_dest), 0)

        # Get the available volume from A --> B --> C
        return min(avail_volume, available_volume_inter_to_dest)

    def _get_stored_indirect_flows(self, src_tor, dest_tor):
        indirect_subflow_ids = self._indirect_subflows_per_rack_pair.get(
            (src_tor, dest_tor),
            [])

        flows = list()
        for subflow_id in indirect_subflow_ids:
            flows.append(self._get_partial_flow_from_subflow_id(subflow_id))

        self._indirect_subflows_per_rack_pair[(src_tor, dest_tor)] = []

        return flows

    def get_remaining_capacity_in_cycle(self, torA, torB):
        """
        Returns the volume that still can be sent in one/the next slot given already known direct and indirect traffic
        Args:
            torA:
            torB:

        Returns:

        """

        direct_demand = self.tor_to_tor_demand_matrix.get_demand_by_node_ids(torA, torB) + \
                        self.tor_to_tor_demand_matrix.get_demand_by_node_ids(torB, torA)
        indirect_demand = \
            self.get_stored_indirect_volume_per_rack_pair(torA, torB) + \
            self.get_stored_indirect_volume_per_rack_pair(torB, torA)
        return max(self.__VOLUME_PER_SLOT - direct_demand - indirect_demand, 0)


@peewee_interface_factory_decorator(interface=AlgorithmInterface)
class RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationConfiguration(AbstractAlgorithmConfiguration):
    """ Configuration class for RotorNet two hop algorithm with full rate allocation."""

    def __init__(self, topology_view_configuration=None,
                 tor2tor_demand_matrix_configuration=None, rate_limit_factor=None,
                 allocate_indirect_flows_strategy_configuration=None):
        """

        Args:
            topology_view_configuration: configuration of topology view, whole topology is used if None
            tor2tor_demand_matrix_configuration: configuration of demand matrix, SimpleTor2TorDemandMatrixWithFlowList
                is used if None
            rate_limit_factor (int): max rate that can be assigned to a flow in a slot multiplied with rate of single
                optical link  (None means all optical links can be used)
        """
        super(RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationConfiguration, self).__init__(
            factory=RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationFactory.__name__
        )
        self.topology_view_configuration = topology_view_configuration
        self.tor2tor_demand_matrix_configuration = tor2tor_demand_matrix_configuration
        self.rate_limit_factor = rate_limit_factor
        self.allocate_indirect_flows_strategy_configuration = allocate_indirect_flows_strategy_configuration
        if self.allocate_indirect_flows_strategy_configuration is None:
            self.allocate_indirect_flows_strategy_configuration = \
                rotornet_strategy.AllToRsAllocateIndirectFlowStrategyConfiguration()

    @property
    def params(self):
        tmp = dict()
        if self.topology_view_configuration is not None:
            tmp["topology_view_configuration"] = self.topology_view_configuration.params
        if self.tor2tor_demand_matrix_configuration is not None:
            tmp["tor2tor-demand_matrix_configuration"] = self.tor2tor_demand_matrix_configuration.params
        if self.rate_limit_factor is not None:
            tmp["rate_limit_factor"] = self.rate_limit_factor
        tmp["indirect_flow_strategy"] = self.allocate_indirect_flows_strategy_configuration.params
        return tmp


@algorithm_factory_decorator
class RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationFactory(AbstractAlgorithmFactory):
    """ Factory class for RotorNet two hop algorithm with full rate allocation."""

    @classmethod
    def generate_algorithm(cls, algorithm_configuration, env):
        import dcflowsim.network.topology
        topo_view = produce_view(algorithm_configuration.topology_view_configuration, env.topology)
        env.topology.add_view(topo_view)
        if algorithm_configuration.tor2tor_demand_matrix_configuration is None:
            dem_matrix = demand_matrix.produce_demand_matrix(
                demand_matrix.SimpleTor2TorDemandMatrixWithFlowListConfiguration(), env)
        else:
            dem_matrix = demand_matrix.produce_demand_matrix(
                algorithm_configuration.tor2tor_demand_matrix_configuration, env)
        env.add_environment_listener(dem_matrix)

        # Calculate max rate per flow capacity * num_all_rotor_links / num_tors (assumes all links have same capacity
        rate_per_link = list(topo_view.ocs_links.values())[0].total_capacity
        if algorithm_configuration.rate_limit_factor is None:
            max_rate = len(topo_view.ocs_links) * rate_per_link / \
                       len(topo_view.get_nodes_by_type(dcflowsim.network.topology.KNOWN_SWITCH_TYPE_TOR_SWITCH))
        else:
            max_rate = rate_per_link * algorithm_configuration.rate_limit_factor

        allocate_indirect_flow_strategy = rotornet_strategy.produce_rotornet_strategy(
            algorithm_configuration.allocate_indirect_flows_strategy_configuration)

        return RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocation(
            env, topo_view, dem_matrix, max_rate,
            allocate_indirect_flows_strategy=allocate_indirect_flow_strategy
        )
