import abc
import logging
from dcflowsim.environment.flow import SuperFlow, FlowListener, ServiceEventCreator
from dcflowsim.utils.number_conversion import *


class EventPriorities:
    EVENT_PRIORITY_SERVICE = 10
    EVENT_PRIORITY_ARRIVAL = 20
    EVENT_PRIORITY_ROTORRECONFIG = 30
    EVENT_PRIORITY_OCSCONFIG = 35
    EVENT_PRIORITY_FINISH = 50


class Event(abc.ABC):
    def __init__(self, event_time):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._event_time = convert_to_decimal(event_time)

    @property
    def event_time(self):
        return self._event_time

    @event_time.setter
    def event_time(self, new_event_time):
        """ Makes sure that the event time is a Decimal according to the definitions.
        This may round the event time to the number of decimal placed defined in
        constants.py"""
        self._event_time = convert_to_decimal(new_event_time)

    @abc.abstractmethod
    def process(self, simulator):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, current_environment_time):
        raise NotImplementedError

    @abc.abstractmethod
    def get_priority(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __lt__(self, other):
        raise NotImplementedError


class ArrivalEvent(Event):
    def __init__(self, arrival_time, flow):
        super(ArrivalEvent, self).__init__(arrival_time)
        self.__flow = flow

    @property
    def flow(self):
        return self.__flow

    def process(self, simulator):
        """
        What I do:
            - we need to add the flow to the system (environment)
            - where do I create the service event? create the service event here

        Args:
            simulator:

        Returns:

        """
        self.logger.debug("Processing flow arrival")
        # First update system
        simulator.environment.advance_system_time(
            self.event_time
        )

        self.__flow.add_listener(
            # Python magic
            ServiceEventCreator(simulator)
        )
        # Second add new flow (hereby we ensure that an algorithm will see flows with current statistics, i.e.,
        # the correct remaining flow volumes
        simulator.environment.add_flow(self.__flow)

        # Add next arrival if generator provides a new flow
        try:
            new_flow = simulator.behavior.traffic_generator.__next__()
            simulator.add_event(
                ArrivalEvent(new_flow.arrival_time, new_flow)
            )
        except StopIteration:
            pass

    def update(self, current_environment_time):
        pass

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_ARRIVAL

    def __str__(self):
        return "ArrivalEvent at {} of flow {}".format(self._event_time, self.__flow)

    def __lt__(self, other):  # <
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        if self.flow.global_id != other.flow.global_id:
            return self.flow.global_id < other.flow.global_id
        raise NotImplementedError("Event comparison not implemented. Event times, priorities and flow IDs are the same")


class ServiceEvent(Event, FlowListener):
    def __init__(self, event_time, flow):
        super(ServiceEvent, self).__init__(event_time)
        self.__flow = flow

    @property
    def flow(self):
        return self.__flow

    def process(self, simulator):
        self.logger.debug("Processing flow  {} service, remaining_volume: {}".format(
            self.__flow.global_id, self.__flow.remaining_volume))
        environment = simulator.environment

        # First update environment
        environment.advance_system_time(
            self.event_time
        )
        # Remove flow from environment
        environment.remove_flow(self.__flow)

        # Event is processed and can be removed as listener from the flow
        self.__flow.remove_listener(self)

    def update(self, current_environment_time):
        if self.__flow.current_rate == 0 and self.__flow.remaining_volume > 0:
            self.event_time = -1
        elif self.__flow.current_rate == 0.0 and self.__flow.remaining_volume == 0.0:  # This flow is another service
            # event following a service event with the same service time......
            pass
        else:
            self.event_time = current_environment_time + self.__flow.remaining_volume / self.__flow.current_rate
        self.logger.debug(
            "New service event of flow {} is at {}".format(
                self.__flow,
                self.event_time
            )
        )

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_SERVICE

    def notify_update_rate(self, flow, current_environment_time):
        self.update(current_environment_time)

    def __lt__(self, other):
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        if self.flow.global_id != other.flow.global_id:
            return self.flow.global_id < other.flow.global_id
        raise NotImplementedError(
            "Event comparison not implemented. Event times, priorities and flow IDs are the same")

    def __str__(self):
        return "ServiceEvent at {} of flow {}".format(self._event_time, self.__flow)


class PartialServiceEvent(Event, FlowListener):
    """ Service event for a partial flow (part of a SuperFlow), Is processed to remove
    a partial flow from the simulation.

    If the volume of the SuperFlow is not transmitted completely, it will spawn and post
    a new PartialServiceEvent to the EventChain. The last PartialServiceEvent will remove
    the corresponding SuperFlow from the system."""

    def __init__(self, event_time, super_flow, corresponding_partial_flow=None):
        assert isinstance(super_flow, SuperFlow), "PartialServiceEvent only usable with SuperFlows"
        super(PartialServiceEvent, self).__init__(event_time)
        self.__super_flow = super_flow
        self.__corresponding_partial_flow = corresponding_partial_flow

    @property
    def super_flow(self):
        return self.__super_flow

    def update(self, current_environment_time):
        """ Update the PartialServiceEvents time so it reflects the next termination of
        a partial flow of the SuperFlow """

        # Get the next time and flow from the superflow logic
        self.event_time, self.__corresponding_partial_flow = \
            self.__super_flow.get_next_expected_partial_service_time_and_flow(
                current_environment_time, self.event_time
            )

    def process(self, simulator):
        """ Process the Partial service event: Check if the SuperFlow/all partial flows
        have finished, if so, remove the SuperFlow, else"""

        if self.event_time > simulator.environment.current_environment_time:
            simulator.environment.advance_system_time(
                self.event_time
            )

        # Current partial flow has finished, remove from topology/free resources
        simulator.environment.release_partial_flow_resources(self.__corresponding_partial_flow)

        simulator.environment.remove_partial_flow(self.__corresponding_partial_flow)

        # Afterwards process the termination of a partial flow in the super_flow
        self.super_flow.process_partial_flow_termination(
            self.__corresponding_partial_flow.global_id,
            self.event_time
        )

        # Check if the SuperFlow is completed, if remove, else spawn new event
        if self.super_flow.transmission_complete():
            simulator.environment.remove_super_flow(self.super_flow)
        else:
            next_time, next_partial_flow = \
                self.super_flow.get_next_expected_partial_service_time_and_flow(self.event_time, self.event_time)
            new_event = PartialServiceEvent(next_time, self.super_flow, next_partial_flow)
            self.super_flow.add_listener(new_event)
            simulator.add_event(new_event)

        # Event is processed and can be removed as listener from the flow
        self.__super_flow.remove_listener(self)

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_SERVICE

    def notify_update_rate(self, flow, current_environment_time):
        self.update(current_environment_time)

    def __lt__(self, other):
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        if self.super_flow.global_id != other.super_flow.global_id:
            return self.super_flow.global_id < other.super_flow.global_id
        raise NotImplementedError(
            "Event comparison not implemented. Event times, priorities and flow IDs are the same")

    def __str__(self):
        return "PartialServiceEvent at {} of superFlow {} for partial flow {}" \
            .format(self._event_time, self.super_flow, self.__corresponding_partial_flow)


class FinishEvent(Event):
    def __init__(self, event_time):
        super(FinishEvent, self).__init__(event_time)

    def process(self, simulator):
        self.logger.debug("Processing simulation finish")
        simulator.stop_simulation = True
        simulator.environment.advance_system_time(self.event_time)

    def update(self, current_environment_time):
        pass

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_FINISH

    def __lt__(self, other):
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        raise NotImplementedError(
            "Event comparison not implemented. Event times and priorities are the same")

    def __str__(self):
        return "FinishEvent at {}".format(self._event_time)


class RotorReconfigEvent(Event):
    def __init__(self, event_time, do_disconnect=False, day_duration=constants.ROTORSWITCH_RECONF_PERIOD,
                 night_duration=constants.ROTORSWITCH_NIGHT_DURATION):
        """
        Reconfigures the RotorSwitches. A new event is inserted to event chain at
        event_time + event_period when the event gets processed.
        Args:
            event_time: time of the event
            event_period: period for the timestamp of the newly inserted event
        """
        super(RotorReconfigEvent, self).__init__(event_time)
        self.day_duration = day_duration
        self.night_duration = night_duration
        self.do_disconnect = do_disconnect

    def process(self, simulator):

        simulator.environment.advance_system_time(self.event_time)
        if self.do_disconnect:
            simulator.environment.disconnect_rotornet_topology()
        else:
            simulator.environment.connect_rotornet_topology()

        if self.do_disconnect:
            self.event_time = simulator.environment.current_environment_time + self.night_duration
            self.do_disconnect = False
        else:
            self.event_time = simulator.environment.current_environment_time + self.day_duration
            self.do_disconnect = True
        simulator.add_event(self)

    def update(self, current_environment_time):
        pass

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_ROTORRECONFIG

    def __lt__(self, other):
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        raise NotImplementedError(
            "Event comparison not implemented. Event times and priorities are the same")

    def __str__(self):
        return "RotorReconfigEvent at {}".format(self._event_time)


class OcsReconfigurationEvent(Event):
    def __init__(self, event_time, ocs, circuits):
        super(OcsReconfigurationEvent, self).__init__(event_time)
        self.__ocs = ocs
        self.__circuits = sorted(circuits)

    @property
    def ocs(self):
        return self.__ocs

    @property
    def circuits(self):
        return self.__circuits

    def process(self, simulator):

        simulator.environment.advance_system_time(self._event_time)

        simulator.environment.connect_optical_circuits(self.__ocs.node_id, self.__circuits)

    def update(self, current_environment_time):
        pass

    def get_priority(self):
        return EventPriorities.EVENT_PRIORITY_OCSCONFIG

    def __lt__(self, other):
        if self.event_time != other.event_time:
            # Special Case: Events with time == -1: put to end
            if self.event_time == -1:
                return False
            if other.event_time == -1:
                return True
            return self.event_time < other.event_time
        if self.get_priority() != other.get_priority():
            return self.get_priority() < other.get_priority()
        if self.ocs != other.ocs:
            return self.ocs.node_id < other.ocs.node_id
        if self.circuits != other.circuits:
            return self.circuits < other.circuits
        raise NotImplementedError(
            "Event comparison not implemented. Event times and priorities are the same")

    def __str__(self):
        return "OcsReconfigurationEvent at {}".format(self._event_time)
