# The Simulations

import abc
from . import event
import logging
from dcflowsim import configuration
from dcflowsim.utils.number_conversion import convert_to_decimal


class AbstractSimulatorConfiguration(configuration.AbstractConfiguration):
    """
    Class representing simulator configuration. Simple configuration.
    """

    def __init__(self, factory):
        super(AbstractSimulatorConfiguration, self).__init__(factory)


class BasicSimulatorConfiguration(AbstractSimulatorConfiguration):
    """
    Class representing a basic simulation configuration.
    """

    def __init__(self, factory, simulation_time, topology_configuration, behavior_configuration,
                 algorithm_configuration,
                 flow_generation_configuration, statistics_configuration, data_model_configuration):
        """
        Init this configuration.

        Args:
            factory: reading this configuration and creating the corresponding object
            simulation_time: to be simulated
            topology_configuration: configuration of the topology
            behavior_configuration: configuration of the simulation behavior
            algorithm_configuration: configuration of the algorithm
            flow_generation_configuration: configuration of the flow generator
            statistics_configuration: many configurations of statistic collecters (a list)
            data_model_configuration: many configurations of data collector (a list)
        """
        super(BasicSimulatorConfiguration, self).__init__(
            factory=factory
        )

        self.simulation_time = convert_to_decimal(simulation_time)
        self.topology_configuration = topology_configuration
        self.behavior_configuration = behavior_configuration
        self.algorithm_configuration = algorithm_configuration
        self.flow_generation_configuration = flow_generation_configuration
        self.statistics_configuration = statistics_configuration
        self.data_model_configuration = data_model_configuration

    @property
    def params(self):
        return self.__dict__


class SimulationComposite(abc.ABC):
    """
    Composite for simulations.
    The composite has a first level leave which is the simulator.
    On the first level there is also a composite called MasterSimulation which
    contains multiple simulators.
    """
    def __init__(self, behavior):
        """
        Constructor.
        """
        self._stop_simulation = False
        self._behavior = behavior

    @property
    def stop_simulation(self):
        return self._stop_simulation

    @stop_simulation.setter
    def stop_simulation(self, stop_value):
        self._stop_simulation = stop_value

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, new_behavior):
        self._behavior = new_behavior

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class Simulator(SimulationComposite):

    """ The simulation. A Leave of the Composite """

    def __init__(self, environment, simulation_time, behavior=None):
        """
        Construct object.

        Args:
            environment(Environment): The simulation environment
            simulation_time(float): The simulation time (duration)
            behavior(AbstractSimulationBehavior): The simulation behavior
        """
        super(Simulator, self).__init__(behavior)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__environment = environment
        self.__event_list = list()
        self.__simulation_time = simulation_time
        self.__init_simulation()

    @property
    def event_list(self):
        return self.__event_list

    @property
    def environment(self):
        return self.__environment

    def __init_simulation(self):
        self.logger.info("Adding FinishEvent to event list")
        self.__event_list.append(
            event.FinishEvent(self.__simulation_time)
        )

    def add_event(self, new_event):
        self.logger.debug("Adding event " + str(new_event))
        self.__event_list.append(new_event)

    def run(self):
        self._behavior.run(self)

    def sort_event_list(self):
        self.__event_list.sort()

    def remove_events_of_type(self, event_type):
        """
        Removes all events of the provided type from the event list
        Args:
            event_type: type of the events that shall be removed

        """
        events_to_keep = list()
        for ev in self.__event_list:
            if not isinstance(ev, event_type):
                events_to_keep.append(ev)
        self.logger.debug("Removed {} of {} events from the event list.".format(
            len(self.__event_list) - len(events_to_keep),
            len(self.__event_list)
        ))
        self.__event_list = events_to_keep

    def reset(self):
        """
        Reset the simulation.
        - reset topology
        - reset flows
        - reset sim time
        - reset listeners
        - reset event list
        - reset behavior.processed_events
        - reset stop_simulation
        - init_simulation

        Returns:

        """
        self.environment.clear()
        self.environment.reset_listeners()
        self.environment.current_environment_time = 0
        self.__event_list = list()
        self.__init_simulation()
        self._stop_simulation = False
