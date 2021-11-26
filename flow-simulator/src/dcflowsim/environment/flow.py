import logging

from dcflowsim.utils.number_conversion import *
from dcflowsim.network.networkx_wrapper import Path
from decimal import Decimal


class FlowListener(object):
    def notify(self, flow_event):
        flow_event.visit(self)

    def notify_update_rate(self, flow, current_environment_time):
        pass

    def notify_first_time_with_rate(self, flow):
        pass

    def notify_update_remaining_volume(self, flow, volume_change):
        pass


class ServiceEventCreator(FlowListener):

    def __init__(self, simulator):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._simulator = simulator

    def notify_first_time_with_rate(self, flow):
        from dcflowsim.simulation.event import PartialServiceEvent, ServiceEvent
        assert flow.current_rate > 0
        # Switch for creation of PartialServiceEvents
        if isinstance(flow, SuperFlow):
            service_time, service_partial_flow = \
                flow.get_next_expected_partial_service_time_and_flow(
                    self._simulator.environment.current_environment_time, None
                )
            new_service = PartialServiceEvent(service_time, flow, service_partial_flow)
            flow.add_listener(new_service)
            self.logger.debug(
                "Creating partial service event at t=%s" % new_service.event_time)

        # Standard service events for normal Flow
        else:
            if flow.current_rate == 0:
                expected_service_time = -1
            else:
                expected_service_time = self._simulator.environment.current_environment_time + \
                                        flow.remaining_volume / flow.current_rate
            # Finally add the service event for this new flow
            new_service = ServiceEvent(
                expected_service_time,
                flow
            )
            flow.add_listener(new_service)
            self.logger.debug(
                "Creating (expected) service event at t=%s" % new_service.event_time)

        self._simulator.add_event(new_service)
        flow.remove_listener(self)


class FlowNotificationEvent(object):
    def visit(self, listener):
        raise NotImplementedError


class FlowRateUpdateNotificationEvent(FlowNotificationEvent):
    def __init__(self, flow, current_environment_time):
        self.current_environment_time = current_environment_time
        self.flow = flow

    def visit(self, listener):
        listener.notify_update_rate(self.flow, self.current_environment_time)


class FlowFirstTimeWithRateNotificationEvent(FlowNotificationEvent):
    def __init__(self, flow):
        self.flow = flow

    def visit(self, listener):
        listener.notify_first_time_with_rate(self.flow)


class FlowRemainingVolumeUpdateNotificationEvent(FlowNotificationEvent):
    def __init__(self, flow, volume_change):
        self.flow = flow
        self.volume_change = volume_change

    def visit(self, listener):
        listener.notify_update_remaining_volume(self.flow, self.volume_change)


class Flow(object):
    def __init__(self, global_id, arrival_time, source, destination, volume):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__global_id = global_id
        self.__arrival_time = convert_to_decimal(arrival_time)
        self.__source = source
        self.__destination = destination
        self.__volume = convert_to_decimal(volume)

        self.__remaining_volume = Decimal(self.__volume)
        self.__current_rate = Decimal('0')
        self.__last_update_time = Decimal(self.__arrival_time)
        self.__network_routes = None
        self.__system_time = Decimal('0')
        self.__first_time_with_rate = None  # First time this flow was assigned rate

        self.__listeners = list()
        # Keep track if we have already a service event created for this flow
        self.__service_event_created = False

    @property
    def system_time(self):
        return self.__system_time

    @system_time.setter
    def system_time(self, current_system_time):
        self.__system_time = Decimal(current_system_time) - self.__arrival_time

    @property
    def first_time_with_rate(self):
        return self.__first_time_with_rate

    @property
    def network_route(self):
        return self.__network_routes

    def set_network_route(self, route, current_environment_time):
        """
        Set the network route of the flow. Triggers notification of listeners
        Args:
            route:
            current_environment_time:

        Returns:

        """
        self.update_rate(route.rate, current_environment_time)
        self.__network_routes = route
        self.notify_listeners(FlowRateUpdateNotificationEvent(self, current_environment_time))
        if not self.__service_event_created and route.rate > Decimal('0'):
            self.notify_listeners(FlowFirstTimeWithRateNotificationEvent(self))

    @property
    def global_id(self):
        return self.__global_id

    @property
    def source(self):
        return self.__source

    @source.setter
    def source(self, source):
        self.__source = source

    @property
    def destination(self):
        return self.__destination

    @destination.setter
    def destination(self, destination):
        self.__destination = destination

    @property
    def arrival_time(self):
        return self.__arrival_time

    @arrival_time.setter
    def arrival_time(self, arrival_time):
        self.__arrival_time = Decimal(arrival_time)

    @property
    def volume(self):
        return Decimal(self.__volume)

    @volume.setter
    def volume(self, volume):
        self.__volume = Decimal(volume)

    @property
    def remaining_volume(self):
        return self.__remaining_volume

    @remaining_volume.setter
    def remaining_volume(self, remaining_volume):
        self.__remaining_volume = convert_to_decimal(remaining_volume)

    @property
    def current_rate(self):
        return Decimal(self.__current_rate)

    @current_rate.setter
    def current_rate(self, current_rate):
        self.__current_rate = convert_to_decimal(current_rate)

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    def notify_listeners(self, notification_event):
        for listener in self.__listeners:
            listener.notify(notification_event)

    def reset_listeners(self):
        self.__listeners = list()

    def update_rate(self, rate, current_environment_time=0):
        """
        Update the flow. Call it with the current update time and the new rate.
        Before updating the attributes, the remaining volume will be determined based on the elapsed time
        since the last update event.

        Args:
            rate: the new rate to be set
            current_environment_time: current time of the environment

        Returns:

        """

        if rate is None:
            raise Exception("Rate is None! Should not happen! Flow.id {} Rate {}".format(
                self.global_id,
                rate
            ))
        self.__current_rate = Decimal(rate)
        self.__last_update_time = current_environment_time
        self.logger.debug(
            "ID: {}: Updated rate to {} bps.".format(
                self.global_id,
                Decimal(self.__current_rate)
            )
        )

    def update_remaining_volume(self, current_update_time):
        delta_time = current_update_time - self.__last_update_time
        if delta_time < 0:
            raise RuntimeError("This should not happen. New update time is smaller than last update time")
        transmitted_volume = round_decimal_up(delta_time * self.__current_rate)

        # Because of rounding errors, the transmitted_volume can be a little bit larger
        # than the remaining_volume. We make sure to never get a negative remaining volume.
        true_volume_to_transmit = min(self.__remaining_volume, transmitted_volume)
        self.remaining_volume = self.__remaining_volume - true_volume_to_transmit

        # Check if this is the first chunk of transmission
        if self.__first_time_with_rate is None and self.__current_rate > Decimal('0'):
            self.__first_time_with_rate = self.__last_update_time

        # Do timing stuff here
        self.__last_update_time = current_update_time
        self.__system_time = current_update_time - self.__arrival_time

        self.logger.debug(
            "ID: {}: {} bits transmitted. New remaining volume is {}".format(
                self.global_id,
                Decimal(transmitted_volume),
                self.__remaining_volume
            )
        )
        self.notify_listeners(FlowRemainingVolumeUpdateNotificationEvent(self, true_volume_to_transmit))

    def reset(self):
        """
        Reset the flow.

        """
        self.__current_rate = Decimal('0')
        self.__network_routes = None
        self.notify_listeners(FlowRateUpdateNotificationEvent(self, None))

    def __str__(self):
        return f"ID: {self.global_id}, {self.source}->{self.destination}, Arrival: {self.arrival_time}, " \
               f"Vol={self.volume}, Remaining Vol={self.remaining_volume}, System Time: {self.system_time}"


class NormalFlowFactory(object):
    """ Factory for creation of a standard flow object"""

    @classmethod
    def produce(cls, global_id, arrival_time, source, destination, volume):
        """ Produce a Flow"""
        return Flow(global_id, arrival_time, source, destination, volume)


class SuperFlow(FlowListener):
    """ Class that encapsulates multiple Subflows between a source and a destination.
    A Subflow is a sequence of Flow objects. It is defined by a list of intermediate nodes.
    Direct routing of the SuperFlow is represented as a Subflow with a single Flow.

    Naming is as follows:
        'SuperFlow' is the equivalent of a demand between source and destination.

        'Subflow' is a sequence of hops the traffic of the SuperFlow should take. Each
        Subflow consists of a sequence of Flow objects that are routed.

        The flows that are actually routed are referred to as 'partial Flow' of the SuperFlow.
            
    Volumes are defined as follows:
        'Remaining volume' has not yet reached the destination node

        'Traveling volume' is volume bound in already created SubFlows

        'Routable volume' is the difference of the above volumes

    Each partial Flow has it's global ID set to an unique string identifier:
        
        <SuperFlowID>_<SubFlowID>_<PartialFlowID>
    
        SuperFlowID is the ID of the parent SuperFlow object
        SubFlowID is a counter per SuperFlow starting at 1
        PartialFlowID is a conter per SubFlow staring at 1

        So "1_3_2" would be the ID of the second Partial Flow of the third SubFlow of the
        Superflow with global ID 1.  
    """

    def __init__(self, global_id, arrival_time, source, destination, volume):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__global_id = global_id
        self.__arrival_time = Decimal(arrival_time)
        self.__source = source
        self.__destination = destination
        self.__volume = Decimal(volume)

        self.__remaining_volume = Decimal(self.__volume)
        self.__last_update_time = Decimal(self.__arrival_time)
        self.__current_travelling_volume = Decimal('0')  # accumulation of volume of partial flows
        self.__active_flows = {}  # subflow_id -> flow, active flows for all subflows
        self.__pending_intermediate_nodes = {}  # subflow_id -> nodes, nodes that still need to be visited by a subflow
        self.__system_time = Decimal('0')
        self.__nr_subflows = 0  # tracks the number of subflows, use to generate subflow_ids.
        self.__flow_id_2_subflow_id = {}  # flow_id --> subflow_id, subflow id is unique for a created sequence of flows.
        self.__first_time_with_rate = None

        self.__listeners = list()
        self.__service_event_created = False

    @property
    def listeners(self):
        return self.__listeners

    @property
    def global_id(self):
        return self.__global_id

    @property
    def arrival_time(self):
        return self.__arrival_time

    @property
    def source(self):
        return self.__source

    @property
    def destination(self):
        return self.__destination

    @property
    def volume(self):
        return Decimal(self.__volume)

    @volume.setter
    def volume(self, volume):
        self.__volume = Decimal(volume)

    @property
    def remaining_volume(self):
        return self.__remaining_volume

    @remaining_volume.setter
    def remaining_volume(self, remaining_volume):
        self.__remaining_volume = Decimal(remaining_volume)

    @property
    def current_travelling_volume(self):
        return self.__current_travelling_volume

    @property
    def active_flows(self):
        return self.__active_flows.values()

    @property
    def system_time(self):
        return self.__system_time

    @system_time.setter
    def system_time(self, current_system_time):
        self.__system_time = Decimal(current_system_time) - self.__arrival_time

    @property
    def first_time_with_rate(self):
        return self.__first_time_with_rate

    @property
    def network_route(self):
        """
        The network route of the only active sub-flow. Only applicable to single path routing case
        Returns:
            the network route object of the sub-flow
        """
        if self.nr_subflows == 0:
            return None
        elif self.nr_subflows == 1:
            return self.active_flows.__iter__().__next__().network_route
        raise RuntimeError("Network route for SuperFlow only supported in single path routing case.")

    def set_network_route(self, route, current_environment_time):
        """
        Setting network route of this flow. Can only be used in single path routing case.
        If it does not already exist, spawns a single sub-flow with the entire volume and assigns the route
        Args:
            route: The network route to be assigned
            current_environment_time: Current time of environment
        Returns:

        """
        assert self.nr_subflows <= 1
        if self.nr_subflows == 0:
            self.logger.debug("Creating new sub-flow to assign network rate.")
            # Sub-flow not created yet. Spawn it with full volume and no intermediate nodes
            partial_flow, _ = self.create_subflow(current_environment_time, self.volume, intermediate_nodes=None)
        else:
            partial_flow = self.active_flows.__iter__().__next__()  # We have already the partial flow
        partial_flow.set_network_route(route, current_environment_time)

    @property
    def nr_subflows(self):
        return self.__nr_subflows

    @property
    def routable_volume(self):
        """ Returns the volume that can still be routed/bound to a SubFlow, 
        is equal to remaining_volume - volume_in_travel

        Returns:
            The volume that can be bound to a SubFlow 
        """
        return round_decimal(self.remaining_volume - self.current_travelling_volume)

    @property
    def current_rate(self):
        """Iterates over the active flow list and returns the total rate allocated to all of the
        associated active flows.

        Returns:
            Rate of the SuperFlow
        """
        return sum(flow.current_rate for flow in self.__active_flows.values())

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    def notify_listeners(self, notification_event):
        for listener in self.__listeners:
            listener.notify(notification_event)

    def reset_listeners(self):
        self.__listeners = list()

    def get_active_flow(self, active_flow_id):
        return self.__active_flows[active_flow_id]

    def get_active_partial_flow_from_subflow_id(self, subflow_id):
        """ Returns the flow that is currently active for a specific subflow given by
        the subflow_id.

        Args:
            subflow_id (str): String identifier for the subflow (chain of flows) of interest

        Returns:
            The Flow object corresponding to the active flow of the subflow. None if not present.
        """
        try:
            return self.__active_flows[subflow_id]
        except KeyError:
            return None

    def create_subflow(self, creation_time, volume, intermediate_nodes=None):
        """ Creates a new sequence of flows from source to destination with intermediate
        stops at the nodes specified by the intermediate nodes list

        Args:
            creation_time: Time of creation of the first flow of the sequence
            volume: The volume of the partial flow
            intermediate_nodes (List of Nodes): List of the intermediate stops

        Returns:
            (first_flow, new_subflow_id)

        Raises:
            RuntimeError: if the requested volume is not available anymore
        """

        if volume == 0:
            raise RuntimeError("Can't create SubFlow with volume zero!")

        if volume > self.routable_volume:
            raise RuntimeError("Can't create SubFlow with volume {}, not enough "
                               "volume remaining in Superflow".format(volume))

        if intermediate_nodes is None:
            intermediate_nodes = []

        # increase subflow_id
        self.__nr_subflows += 1

        # create the subflow_id and the flow_id for the first flow on the new subflow
        new_subflow_id = str(self.global_id) + "_" + str(self.nr_subflows)
        new_flow_id = new_subflow_id + "_1"

        # append destination of SuperFlow to list of nodes to visit.
        intermediate_nodes.append(self.destination)
        destination_node = intermediate_nodes.pop(0)
        new_active_flow = Flow(new_flow_id, creation_time, self.source,
                               destination_node, volume)
        new_active_flow.add_listener(self)

        self.__active_flows[new_subflow_id] = new_active_flow
        self.__flow_id_2_subflow_id[new_flow_id] = new_subflow_id
        self.__pending_intermediate_nodes[new_subflow_id] = intermediate_nodes
        self.__current_travelling_volume += volume

        return new_active_flow, new_subflow_id

    def update_remaining_volume(self, current_update_time):
        """ Updates the volumes of all active flows

        Args:
            current_update_time: Time used to calculate the volume based on the rates.
        """

        # Update System time for SuperFlow
        self.__system_time = Decimal(current_update_time) - Decimal(self.__arrival_time)

        if self.__first_time_with_rate is None and self.current_rate > Decimal('0'):
            self.__first_time_with_rate = self.__last_update_time

        for partial_flow in self.active_flows:
            partial_flow.update_remaining_volume(current_update_time)

        self.__last_update_time = current_update_time
        # We do not notify listeners here since SuperFlows remaining volume decreases only when partial flows finish.

    def get_next_expected_partial_service_time_and_flow(self, current_environment_time, current_service_time=None):
        """ Calculates the expected service times for all active flows.

        Returns: The next expected partial service time and the flow that will

        """
        relevant_times_to_flow_map = {}

        # loop over all flows, if they have a relevant service time (other than -1)
        # append to find min afterwards.
        for partial_flow in self.active_flows:
            if partial_flow.current_rate == 0 and partial_flow.remaining_volume > 0:
                time_for_current_flow = -1
            # This flow is another service event following a service event
            # with the same service time...
            elif partial_flow.current_rate == 0.0 and partial_flow.remaining_volume == 0.0:
                if current_environment_time is not None:
                    time_for_current_flow = current_environment_time
                else:
                    time_for_current_flow = current_service_time

            # Normal case, calculate exp service time based on rate and remaining volume
            else:
                time_for_current_flow = \
                    current_environment_time + (partial_flow.remaining_volume / partial_flow.current_rate)

            # check if we have flows with times that are not infinity
            if time_for_current_flow != -1:
                relevant_times_to_flow_map[time_for_current_flow] = partial_flow

        # check if relevant flows other that not routed ones are present
        if relevant_times_to_flow_map:
            min_time = min(list(relevant_times_to_flow_map.keys()))
            return min_time, relevant_times_to_flow_map[min_time]
        else:
            return -1, None

    def process_partial_flow_termination(self, partial_flow_id, current_environment_time):
        """ Processes the termination of a partial flow. Spawns a new active partial
        flow if needed to complete the sequence to the destination node

        Args:
            partial_flow_id: Global ID of the partial flow that terminated
            current_environment_time: Time when the partial flow terminated

        Returns:
            new_partial_flow_id: The ID of the newly created flow, or None if no creation
        """

        # Get subflow_id from old_flow
        subflow_id = self.__flow_id_2_subflow_id[partial_flow_id]

        # Get the corresponding Flow object
        old_partial_flow = self.__active_flows[subflow_id]

        assert old_partial_flow.remaining_volume == 0, "Remaining volume of partial flow must be zero"

        # Setup return value
        new_partial_flow = None

        # We need to spawn a new flow to another node, last one is SuperFlow dest.
        if self.__pending_intermediate_nodes[subflow_id]:
            # Get the current id of the partial flow, partial_flow_id is of form
            # "<SuperFlowID>_<SubflowID>_<PartialFlowID>"
            new_sub_id = int(partial_flow_id.split("_")[2]) + 1
            new_partial_flow_id = subflow_id + "_" + str(new_sub_id)

            # create the next flow, from former destination to next intermediate node
            new_source = old_partial_flow.destination
            new_destination = self.__pending_intermediate_nodes[subflow_id].pop(0)
            new_partial_flow = Flow(new_partial_flow_id, current_environment_time, new_source,
                                    new_destination, old_partial_flow.volume)
            new_partial_flow.add_listener(self)

            # update the active flow and pending intermedate nodes dicts
            self.__active_flows[subflow_id] = new_partial_flow
            self.__flow_id_2_subflow_id[new_partial_flow_id] = subflow_id

        # We do not have to spawn a new flow, the partial flow has reached the dest node
        # the Subflow is therefore complete.
        else:
            # reduce the volume that is bound in partial flows
            self.__current_travelling_volume -= old_partial_flow.volume
            # reduce the SuperFlow volume
            self.__remaining_volume -= old_partial_flow.volume
            # delete the empty list of intermediate nodes
            del self.__pending_intermediate_nodes[subflow_id]
            del self.__active_flows[subflow_id]

            # Change volume is that of finished sub-flow
            self.notify_listeners(FlowRemainingVolumeUpdateNotificationEvent(self, old_partial_flow.volume))

        # Delete the partial flow
        old_partial_flow.remove_listener(self)
        del self.__flow_id_2_subflow_id[partial_flow_id]

        return new_partial_flow

    def transmission_complete(self):
        """ Checks if the transmission is complete:
        Remaining volume is 0, No active partial flows, no volume in travel anymore"""
        complete = self.__remaining_volume == Decimal('0')
        complete = complete and not list(self.active_flows)
        complete = complete and self.__current_travelling_volume == Decimal('0')
        return complete

    def reset(self):
        """ Resets all active partial flows"""

        for partial_flow in self.active_flows:
            partial_flow.reset()
        self.notify_listeners(FlowRateUpdateNotificationEvent(self, None))

    def notify_update_rate(self, flow, current_environment_time):
        self.notify_listeners(FlowRateUpdateNotificationEvent(self, current_environment_time))

    def notify_first_time_with_rate(self, flow):
        if self.__service_event_created:
            return
        self.notify_listeners(FlowFirstTimeWithRateNotificationEvent(self))

    def __str__(self):
        return f"ID: {self.global_id}, {self.source}->{self.destination}, Arrival: {self.arrival_time}, " \
               f"Vol={self.volume}, Remaining Vol={self.remaining_volume}, System Time: {self.system_time}"


class SuperFlowFactory(object):
    """ Factory class for SuperFlows"""

    @classmethod
    def produce(cls, global_id, arrival_time, source, destination, volume):
        return SuperFlow(global_id, arrival_time, source, destination, volume)


class Route(object):
    def __init__(self, rate, links):
        """
        Init a route with a rate and the links

        Args:
            rate: of the route
            links: a list with tuples specifying the links.
                example: links = [(1,2),(2,3)] will be a route from 1 - 2 - 3
        """
        decimal_rate = convert_to_decimal(rate)
        decimal_rate_rounded = round_decimal(decimal_rate)

        # A Rate of higher precision than the links should be rejected
        assert decimal_rate == decimal_rate_rounded, \
            "The given rate {} has higher precision than the link capacities " \
            "(max {} decimal places). Make sure to round them properly in the " \
            "algorithm.".format(decimal_rate, constants.FLOAT_DECIMALS)

        self.__rate = decimal_rate_rounded
        self.__links = links

    @property
    def rate(self):
        return self.__rate

    @property
    def links(self):
        """
        Provides a list with the link tuple objects. Link tuple specifies an edge via its node ids.

        Returns:
            a list with the link tuple objects.

        """
        return self.__links

    @property
    def path_object(self):
        """ Reconstruct a Path object from the Route. """
        if len(self.links) == 0:
            return Path(list())
        if len(self.links[0]) == 2:  # Case for default links (src, dest)
            return Path(self.path)
        else:  # Case for links with identifier
            tmp_path = []
            tmp_link_identifiers = []
            for src, dest, ident in self.links:
                tmp_path.append(src)
                tmp_link_identifiers.append(ident)
            tmp_path.append(self.links[-1][1])
            return Path(tmp_path, tmp_link_identifiers)

    @property
    def path(self):
        """
        Reconstruct the path from the link list.

        Returns:
            a list with node ids as paths.

        """
        if len(self.links) == 0:
            return []
        path = [link[0] for link in self.links]
        path.append(self.links[-1][1])

        return path

    def __str__(self):
        return "Rate={}, Used Links: {}".format(
            self.rate,
            self.links
        )


class FlowIdGenerator(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.flow_id_pool = set()
        self.flow_id_counter = 0

    def get_new_id(self):
        if len(self.flow_id_pool) == 0:
            tmp_id = self.flow_id_counter
            self.flow_id_counter += 1
            return tmp_id
        return self.flow_id_pool.pop()

    def return_id(self, flow_id):
        self.flow_id_pool.add(flow_id)
