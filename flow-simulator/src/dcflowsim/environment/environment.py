import abc
import logging
from sortedcontainers import SortedList
from decimal import Decimal

from . import flow
from dcflowsim.network import network_elements, ocs_topology


class EnvironmentUpdateEvent(abc.ABC):
    """
    Abstract environment update event class.

    Every event changing the environment should inherit from this class.

    """

    @abc.abstractmethod
    def visit(self, listener):
        raise NotImplementedError


class EnvironmentFlowAddEvent(EnvironmentUpdateEvent):

    def __init__(self, flow):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.new_flow = flow

    def visit(self, listener):
        listener.notify_add_flow(self.new_flow)

    def __str__(self):
        return "EnvironmentFlowAddEvent"


class EnvironmentFlowRemoveEvent(EnvironmentUpdateEvent):
    def __init__(self, removed_flow):
        self.removed_flow = removed_flow

    def visit(self, listener):
        listener.notify_rem_flow(self.removed_flow)

    def __str__(self):
        return "EnvironmentFlowRemoveEvent"


class EnvironmentPartialFlowRemoveEvent(EnvironmentUpdateEvent):
    def __init__(self, removed_flow):
        self.removed_flow = removed_flow

    def visit(self, listener):
        listener.notify_partial_rem_flow(self.removed_flow)

    def __str__(self):
        return "EnvironmentPartialFlowRemoveEvent"


class EnvironmentTimeUpdateEvent(EnvironmentUpdateEvent):
    def visit(self, listener):
        listener.notify_upd_time()

    def __str__(self):
        return "EnvironmentTimeUpdateEvent"


class EnvironmentFlowRouteUpdateEvent(EnvironmentUpdateEvent):
    def __init__(self, changed_flow):
        self.changed_flow = changed_flow

    def visit(self, listener):
        listener.notify_upd_flow_route(self.changed_flow)

    def __str__(self):
        return "EnvironmentFlowRouteEvent"


class EnvironmentOcsUpdateEvent(EnvironmentUpdateEvent):
    def __init__(self, ocs):
        self.updated_ocs = ocs

    def visit(self, listener):
        listener.notify_upd_ocs(self.updated_ocs)

    def __str__(self):
        return "EnvironmentOcsUpdateEvent"


class EnvironmentResetEvent(EnvironmentTimeUpdateEvent):
    def visit(self, listener):
        listener.notify_reset()

    def __str__(self):
        return "EnvironmentResetEvent"


class EnvironmentOcsUpdateRequestEvent(EnvironmentUpdateEvent):
    def __init__(self, ocs, circuits):
        self.update_requested_ocs = ocs
        self.circuits = circuits

    def visit(self, listener):
        listener.notify_upd_req_ocs(self.update_requested_ocs, self.circuits)

    def __str__(self):
        return "EnvironmentOcsUpdateRequestEvent"


class Environment(flow.FlowListener):
    """
    The environment class.

    """

    def __init__(self, topology):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__current_environment_time = Decimal('0.0')
        self.__topology = topology
        self.__flows = dict()  # flow id -> object
        self.flow_id_generator = flow.FlowIdGenerator()
        self.__environment_listeners = SortedList()

        self.__active_flows = set()

    def notify_update_rate(self, updated_flow, current_environment_time):
        if updated_flow.current_rate > Decimal('0'):
            self.__active_flows.add(updated_flow)
        elif updated_flow.current_rate == Decimal('0'):
            if updated_flow in self.__active_flows:
                self.__active_flows.remove(updated_flow)

    def empty(self):
        return len(self.__flows) == 0

    def add_environment_listener(self, listener):
        self.__environment_listeners.add(listener)
        self.logger.info(f"Added environment listener {listener}")

    def remove_environment_listener(self, listener):
        """
        Removes an already registered listener

        Args:
            listener: EnvironmentListener to be removed

        Returns:

        """
        self.__environment_listeners.remove(listener)
        self.logger.info(f"Removed environment listener {listener}")

    @property
    def listeners(self):
        return self.__environment_listeners

    def notify_listeners(self, notification_event):
        for listener in self.__environment_listeners:
            self.logger.debug(f"Notifying {listener}. Event is {notification_event}")
            listener.notify(notification_event)

    @property
    def topology(self):
        return self.__topology

    @property
    def flows(self):
        return self.__flows

    @property
    def current_environment_time(self):
        return self.__current_environment_time

    @current_environment_time.setter
    def current_environment_time(self, new_time):
        self.__current_environment_time = new_time

    def set_flows(self, flows):
        """
        Replace flow container of environment. ONLY FOR INITIALIZATION PURPOSE
        Args:
            flows:

        Returns:

        """
        self.__flows = flows

    def get_flow_by_id(self, global_flow_id):
        """ Get a Flow from the environment by the global ID

        Returns:
            The Flow object with the ID, None if not present
        """
        try:
            return self.__flows[global_flow_id]
        except KeyError:
            return None

    def add_flow(self, flow):
        self.__flows[flow.global_id] = flow
        flow.add_listener(self)
        self.logger.debug("Added flow.")
        self.notify_listeners(EnvironmentFlowAddEvent(flow))

    def remove_flow(self, flow):
        assert flow.remaining_volume == Decimal('0.0'), \
            "Should not remove flow with non zero remaining volume!"
        self.__free_resources_of_flow(flow)
        del self.__flows[flow.global_id]
        if flow in self.__active_flows:
            self.__active_flows.remove(flow)
        flow.remove_listener(self)
        self.logger.debug("Removed flow.")
        self.notify_listeners(EnvironmentFlowRemoveEvent(flow))

    def release_partial_flow_resources(self, partial_flow):
        """ Release the resources of a partial flow"""
        self.__free_resources_of_flow(partial_flow)
        self.logger.debug("Released partial flow resources")

    def remove_super_flow(self, super_flow):
        """ Remove a superflow from the simulation, no resources need to be released"""
        del self.__flows[super_flow.global_id]
        if flow in self.__active_flows:
            self.__active_flows.remove(super_flow)
        super_flow.remove_listener(self)
        self.logger.debug("Removed SuperFlow.")
        self.notify_listeners(EnvironmentFlowRemoveEvent(super_flow))

    def remove_partial_flow(self, partial_flow):
        """Only notifies listeners that a partial_flow has been removed."""
        self.notify_listeners(EnvironmentPartialFlowRemoveEvent(partial_flow))

    def advance_system_time(self, current_system_time):
        """
        When do we have to react?
        - react when flows were removed
        - react when flows were added
        - react when simulation finishes

        Returns:
            None
        """
        self.logger.debug(f"Advancing time from {self.current_environment_time} to {current_system_time}")
        self.__update_flows(current_system_time)
        self.__current_environment_time = current_system_time

        self.notify_listeners(EnvironmentTimeUpdateEvent())

    def __update_flows(self, current_system_time):
        """
        Update the volume of the flows
        Args:
            delta_time:

        Returns:

        """
        self.logger.debug("Advancing flow transmissions by {}".format(
            current_system_time - self.current_environment_time
        ))
        # for flow in self.__flows.values():
        for flow in self.__active_flows:
            flow.update_remaining_volume(
                current_system_time
            )

    def reset(self):
        """
        Simply resets everything, i.e., any configurations, flows etc...

        """
        self.reset_flows()
        self.reset_topology()

    def reset_flow(self, flow_obj):
        """
        Method to reset a given flow in the environment. Flow rates are set to zero and
        flow routes are set to None. Also the used resources on the topology are released
        """
        if type(flow_obj) == flow.Flow:
            if flow_obj.network_route is not None:
                self.__free_resources_of_flow(flow_obj)
            flow_obj.reset()
        elif type(flow_obj) == flow.SuperFlow:
            for partial_flow in flow_obj.active_flows:
                if partial_flow.network_route is not None:
                    self.release_partial_flow_resources(partial_flow)
                partial_flow.reset()

    def reset_flows(self):
        """
        Iterates through all the flows in the environment and resets them
        """

        for active_flow in set(self.__active_flows):
            self.reset_flow(active_flow)

    def reset_topology(self):
        """
        Method resets the topology. Mostly affects optical topologies.

        """
        self.topology.reset()

    def clear(self):
        """
        Resets the topology and REMOVES all flows from the environment (flow list is empty afterwards)

        Returns:

        """
        self.reset_flows()
        self.reset_topology()
        self.__flows = dict()

    def reset_listeners(self):
        """ Trigger EnvironmentResetEvent for all listeners """
        self.notify_listeners(EnvironmentResetEvent())

    def __free_resources_of_flow(self, flow):
        self.logger.debug(f"Freeing resources (capacity) of flow {flow}")
        self.__topology.release_route(flow)

    def allocate_flow(self, new_flow, path, rate):
        """
        Allocates a flow on the given route. Notifies listeners, if the flow had another route before and the route is
        different now
        Args:
            new_flow: is the flow
            path: is a list with node ids?!
            rate: the rate to be allocated

        """
        old_route = new_flow.network_route
        new_flow.set_network_route(
            flow.Route(
                rate,
                path.links
            ),
            self.current_environment_time
        )

        # Allocate the flow on the 'original' topology ...
        self.__topology.allocate_route(new_flow)
        self.logger.info(
            "Allocated flow {} route {}".format(
                new_flow,
                new_flow.network_route
            )
        )
        if old_route is not None and old_route.links != new_flow.network_route.links:
            self.notify_listeners(EnvironmentFlowRouteUpdateEvent(new_flow))

    def __check_and_release_existing_optical_circuit(self, ocs, node):
        other_ep_id = ocs.get_existing_circuit(node.node_id)
        if other_ep_id is not None:
            self.topology.reset_circuit(ocs, node, self.topology.get_node(other_ep_id))

    def release_optical_circuits(self, ocs_id, circuits):
        """
        Release the given sets of circuits for the given ocs. No reconfiguration delay (we have it when new circuits
        are set). All flows should already be removed.
        Args:
            ocs_id (str): Node ID of the OCS to work on
            circuits (list): list of tuples containing the ids of the nodes that shall be disconnected

        Returns:

        """
        if not isinstance(self.topology, ocs_topology.TopologyWithOcs):
            self.logger.warning("Requested to set circuit but topology has no OCS.")
            return
        # Just to be sure: Check if we have a "normal" OCS and not a RotorSwitch
        ocs = self.topology.get_node(ocs_id)
        assert type(ocs) == network_elements.OpticalCircuitSwitch
        for node1_id, node2_id in circuits:
            node1 = self.topology.get_node(node1_id)
            node2 = self.topology.get_node(node2_id)
            self.logger.debug(f"Release circuit between {node1} and {node2} ad {ocs}")
            self.topology.reset_circuit(ocs, node1, node2)

    def set_optical_circuits(self, ocs_id, circuits):
        """
        Sets the circuits given in the list at the given OCS. Then notifies listeners that there was a potential
        topology change. If the topology does not have an OCS, nothing happens.
        Existing conflicting circuits are released.
        Args:
            ocs_id (str): Node ID of the OCS to work on
            circuits (list): list of tuples containing the ids of the nodes that shall be connected

        Returns:

        """
        # Just to be sure: Check if we have a "normal" OCS and not a RotorSwitch
        if not isinstance(self.topology, ocs_topology.TopologyWithOcs):
            self.logger.warning("Requested to set circuit but topology has no OCS.")
            return
        # Just to be sure: Check if we have a "normal" OCS and not a RotorSwitch
        ocs = self.topology.get_node(ocs_id)
        assert type(ocs) == network_elements.OpticalCircuitSwitch

        # Release and directly set the circuit if there is no reconfig delay. This avoid extra event scheduling.
        # If ocs has reconfig delay, release the circuit and schedule an event to set the circuit in the future.

        if ocs.reconfig_delay == 0:
            for node1_id, node2_id in circuits:
                node1 = self.topology.get_node(node1_id)
                node2 = self.topology.get_node(node2_id)
                # Check if there is already a circuit
                self.__check_and_release_existing_optical_circuit(ocs, node1)
                self.__check_and_release_existing_optical_circuit(ocs, node2)

                self.logger.debug("Setting circuit between {} and {} at {}".format(
                    node1, node2, ocs
                ))
                self.topology.set_circuit(ocs, node1, node2)

            # Notify listeners:
            self.notify_listeners(EnvironmentOcsUpdateEvent(ocs))
        else:
            for node1_id, node2_id in circuits:
                node1 = self.topology.get_node(node1_id)
                node2 = self.topology.get_node(node2_id)
                self.logger.debug("Resetting circuit between {} and {} at {}".format(
                    node1, node2, ocs
                ))

                # Check if there is already a circuit
                self.__check_and_release_existing_optical_circuit(ocs, node1)
                self.__check_and_release_existing_optical_circuit(ocs, node2)

            self.notify_listeners(EnvironmentOcsUpdateRequestEvent(ocs, circuits))

    def connect_optical_circuits(self, ocs_id, circuits):
        """
        Sets the given circuits at the OCS with the given ID
        Args:
            ocs_id (str): Node ID of OCS
            circuits (list): list of tuples of node ids (strings)

        Returns:

        """
        if not isinstance(self.topology, ocs_topology.TopologyWithOcs):
            self.logger.warning("Requested to set circuit but topology has no OCS.")
            return
        # Just to be sure: Check if we have a "normal" OCS and not a RotorSwitch
        ocs = self.topology.get_node(ocs_id)
        assert type(ocs) == network_elements.OpticalCircuitSwitch

        for node1_id, node2_id in circuits:
            node1 = self.topology.get_node(node1_id)
            node2 = self.topology.get_node(node2_id)
            self.logger.debug("Setting circuit between {} and {} at {}".format(
                node1, node2, ocs
            ))

            self.topology.set_circuit(ocs, node1, node2)

        # Notify listeners:
        self.notify_listeners(EnvironmentOcsUpdateEvent(ocs))

    def disconnect_rotornet_topology(self):
        for rotor_switch in self.topology.get_nodes_by_type(network_elements.RotorSwitch.__name__):
            self.topology.disconnect_rotor_matching(rotor_switch)

    def connect_rotornet_topology(self):
        for rotor_switch in self.topology.get_nodes_by_type(network_elements.RotorSwitch.__name__):
            self.topology.connect_rotor_matching(rotor_switch)
            self.notify_listeners(EnvironmentOcsUpdateEvent(rotor_switch))
