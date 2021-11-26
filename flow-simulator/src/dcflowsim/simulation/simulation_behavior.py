import abc
import logging
import time
from git import Repo

from . import event
from .simulation_behavior_factory import simulation_behavior_factory_decorator, produce
from dcflowsim import configuration, constants
from dcflowsim.data_writer import connection_manager, factory as dw_factory, peewee_interface
from dcflowsim.statistic_collector import base_statistic_collector


class SimulationBehaviorConfiguration(configuration.AbstractConfiguration):

    def __init__(self, factory):
        super(SimulationBehaviorConfiguration, self).__init__(factory)

    @property
    def params(self):
        return {}


class AbstractSimulationBehavior(abc.ABC):
    """
    Classes inheriting this abstract class should implement simulation behaviors.
    """

    @abc.abstractmethod
    def run(self, simulator):
        raise NotImplementedError


class SimpleSimulationBehavior(AbstractSimulationBehavior):
    """
    Class implementing a simple simulation behavior. This behavior just generates all flows
    and then starts the simulation.
    """

    def __init__(self, traffic_generator):
        self.traffic_generator = traffic_generator
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.processed_events = 0

    def create_first_events(self, simulator):
        self.logger.info("Generate flow arrival events for simulator: {}".format(simulator))
        cnt = 0
        for flow in self.traffic_generator:
            simulator.add_event(
                event.ArrivalEvent(flow.arrival_time, flow)
            )
            cnt += 1
        simulator.sort_event_list()
        self.logger.info(f"Added {cnt} arrival events.")

    def run(self, simulator):
        """
        Execute simulation from here. This involves traffic generation, data writing, etc...

        All flow arrival events are generated first. Might consume too much memory. Check!
        """
        self.create_first_events(simulator)
        self.logger.info("Simulator started: {}".format(simulator))

        while len(simulator.event_list) > 0 and not simulator.stop_simulation:
            next_event = simulator.event_list.pop(0)

            # Here it should be guaranteed that after the process finishes, the system is in a consinstent state, i.e.:
            # - all flows have a correct rate and remaining volume
            # - we need to update the service times and resort our event list ...
            next_event.process(simulator)

            # Resort event list.
            simulator.sort_event_list()

            self.processed_events += 1
            if self.processed_events % constants.SIMULATION_STATUS_MESSAGE_FREQUENCY == 0:
                self.logger.info(
                    "Processed {} events. {} events remaining".format(
                        self.processed_events,
                        len(simulator.event_list)
                    )
                )

        self.logger.info("Simulation done: {}".format(simulator))


class SimpleSimulationBehaviorConfiguration(configuration.AbstractConfiguration):

    def __init__(self):
        super(SimpleSimulationBehaviorConfiguration, self).__init__(
            factory=SimpleSimulationBehaviorFactory.__name__
        )

    @property
    def params(self):
        return {}


@simulation_behavior_factory_decorator
class SimpleSimulationBehaviorFactory(object):
    @classmethod
    def generate_behavior(cls, behavior_configuration, simulation_configuration, traffic_generator):
        return SimpleSimulationBehavior(traffic_generator)


class SimulationBehaviorSparseArrivalsConfiguration(configuration.AbstractConfiguration):

    def __init__(self):
        super(SimulationBehaviorSparseArrivalsConfiguration, self).__init__(
            factory=SimpleSimulationBehaviorSparseArrivalsFactory.__name__
        )

    @property
    def params(self):
        return {}


class SimpleSimulationBehaviorSparseArrivals(SimpleSimulationBehavior):
    """
    Class implementing a simple simulation behavior. This behavior just generates all flows
    and then starts the simulation.
    """

    def __init__(self, traffic_generator):
        super(SimpleSimulationBehaviorSparseArrivals, self).__init__(traffic_generator)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def create_first_events(self, simulator):
        self.logger.info("Generate flow arrival events for simulator: {}".format(simulator))
        first_flow = self.traffic_generator.__next__()
        simulator.add_event(
            event.ArrivalEvent(first_flow.arrival_time, first_flow)
        )
        simulator.sort_event_list()
        self.logger.info(f"Added initial arrival event.")


@simulation_behavior_factory_decorator
class SimpleSimulationBehaviorSparseArrivalsFactory(object):
    @classmethod
    def generate_behavior(cls, behavior_configuration, simulation_configuration, traffic_generator):
        return SimpleSimulationBehaviorSparseArrivals(traffic_generator)


class SimulationBehaviorWithDatabaseCheckConfiguration(configuration.AbstractConfiguration):
    """
    Configuration for SimulationBehaviorWithDatabaseCheck decorates another SimulationBehaviorConfiguration
    """

    def __init__(self, decorated_behavior, db_config):
        """

        Args:
            decorated_behavior: simulation behavior configuration that is decorated by
                SimulationBehaviorWithDatabaseCheck
            db_config:
        """
        super(SimulationBehaviorWithDatabaseCheckConfiguration, self).__init__(
            SimulationBehaviorWithDatabaseCheckFactory.__name__)
        self.db_config = db_config
        self.decorated_behavior = decorated_behavior

    @property
    def params(self):
        return {
            'decorated_behavior_name': self.decorated_behavior.factory,
            'decorated_behavior_params': self.decorated_behavior.params
        }


class SimulationBehaviorWithDatabaseCheck(AbstractSimulationBehavior):
    """
    Wraps simulation behavior and adds check if simulation has already been done to the routine
    """

    def __init__(self, decorated_sim_behavior, db_interface, simulation_id):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__decorated_sim_behavior = decorated_sim_behavior
        self.__interface = db_interface
        self.__simulation_id = simulation_id

    @property
    def traffic_generator(self):
        return self.__decorated_sim_behavior.traffic_generator

    def run(self, simulator):
        """
        First checks if the simulation has already been done. Then runs the simulation according to the decorated
        behavior. Afterwards, statistics collectors notify their data writers. Finally, the simulation finished
        flag in the database is updated.
        Args:
            simulator:

        Returns:

        """
        t_start = time.time()

        if self.__interface.is_finished(self.__simulation_id):
            self.logger.error("Simulation {} has already been done.".format(self.__simulation_id))
            return

        self.__interface.insert_start_time(self.__simulation_id, t_start)

        self.__interface.insert_commit_sha(simulation_id=self.__simulation_id,
                                           sha=Repo(search_parent_directories=True).head.commit.hexsha)

        self.__decorated_sim_behavior.run(simulator)

        t_end_sim = time.time()
        self.__interface.insert_end_time(self.__simulation_id, t_end_sim)

        # Iterate over all environment listeners and trigger a final write
        for env_listener in simulator.environment.listeners:
            if isinstance(env_listener, base_statistic_collector.AbstractStatisticCollector):
                # this listener is a statistics collector. To make sure that all data is written to make a notification
                # with immediate flag.
                env_listener.notification_behavior.notify(
                    env_listener.listeners,
                    base_statistic_collector.StatisticsWriteSingleEvent(env_listener),
                    immediate=True
                )

        self.__interface.set_finished(self.__simulation_id)
        t_end = time.time()
        self.logger.error(
            "Processing all events took {}s. Total wall clock time is {}s.".format(
                t_end_sim - t_start,
                t_end - t_start
            )
        )


@simulation_behavior_factory_decorator
class SimulationBehaviorWithDatabaseCheckFactory(object):
    @classmethod
    def generate_behavior(cls, behavior_configuration, simulation_configuration, traffic_generator):
        assert isinstance(behavior_configuration, SimulationBehaviorWithDatabaseCheckConfiguration)
        con_man = connection_manager.DatabaseConnectionManager(
            simulation_configuration.behavior_configuration.db_config)
        flowgen_id = dw_factory.produce_interface(
            simulation_configuration.flow_generation_configuration, con_man).save(
            simulation_configuration.flow_generation_configuration
        )
        algo_id = dw_factory.produce_interface(
            simulation_configuration.algorithm_configuration, con_man).save(
            simulation_configuration.algorithm_configuration
        )
        topology_id = dw_factory.produce_interface(
            simulation_configuration.topology_configuration, con_man).save(
            simulation_configuration.topology_configuration
        )
        db_interface = peewee_interface.SimulationInterface(con_man.connection)
        simulation_id = db_interface.create_or_get_simulation_id(
            flowgen_id, algo_id, topology_id,
            simulation_configuration.simulation_time,
            simulation_configuration.behavior_configuration,
            simulation_configuration.factory.__name__
        )

        return SimulationBehaviorWithDatabaseCheck(
            decorated_sim_behavior=produce(
                behavior_configuration.decorated_behavior,
                simulation_configuration,
                traffic_generator
            ),
            db_interface=db_interface,
            simulation_id=simulation_id
        )


class RotornetSimulationBehavior(AbstractSimulationBehavior):
    """
    Class implementing a simulation behavior for rotornet topology simulations. This behavior is similar to
    SimpleSimulationBehavior but inserts a RotorReconfigEvent at TimeStamp 1.
    """

    def __init__(self, traffic_generator, day_duration, night_duration=constants.ROTORSWITCH_NIGHT_DURATION):
        self.traffic_generator = traffic_generator
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.processed_events = 0

        self.day_duration = day_duration
        self.night_duration = night_duration

    def _create_first_events(self, simulator):
        self.logger.info("Generate flow arrival events for simulator: {}".format(simulator))

        for flow in self.traffic_generator:
            simulator.add_event(
                event.ArrivalEvent(flow.arrival_time, flow)
            )
        # First event after one period
        simulator.add_event(event.RotorReconfigEvent(event_time=0, day_duration=self.day_duration,
                                                     do_disconnect=False, night_duration=self.night_duration))
        simulator.sort_event_list()

        self.logger.info("Simulator started: {}".format(simulator))

    def run(self, simulator):
        """
        Execute simulation from here.
        All flow arrival events are generated first.
        Initial RotorReconfigEvent is inserted into the event list.
        """
        self._create_first_events(simulator)

        while len(simulator.event_list) > 0 and not simulator.stop_simulation:
            next_event = simulator.event_list.pop(0)

            # Here it should be guaranteed that after the process finishes, the system is in a consinstent state, i.e.:
            # - all flows have a correct rate and remaining volume
            # - we need to update the service times and resort our event list ...
            next_event.process(simulator)

            # Resort event list.
            simulator.sort_event_list()

            self.processed_events += 1
            if self.processed_events % constants.SIMULATION_STATUS_MESSAGE_FREQUENCY == 0:
                self.logger.info(
                    f"Processed {self.processed_events} events. {len(simulator.event_list)} events remaining. "
                    f"Simtime: {simulator.environment.current_environment_time}"
                )

            # stop simulation if there are no more flows to process
            if len(simulator.event_list) == 2 and simulator.environment.empty():
                for ev in simulator.event_list:
                    if isinstance(ev, event.RotorReconfigEvent):
                        simulator.stop_simulation = True

        self.logger.info("Simulation done: {}".format(simulator))


class RotornetSimulationBehaviorConfiguration(SimulationBehaviorConfiguration):
    """
    Configuration of Rotornet Simulation Behavior
    """

    def __init__(self, day_duration=constants.ROTORSWITCH_RECONF_PERIOD,
                 night_duration=constants.ROTORSWITCH_NIGHT_DURATION):
        """
        Construct object.
        """
        super(RotornetSimulationBehaviorConfiguration, self).__init__(
            factory=RotornetSimulationBehaviorFactory.__name__
        )
        self.day_duration = day_duration
        self.night_duration = night_duration

    @property
    def params(self):
        return {
            'day_duration': self.day_duration,
            'night_duration': self.night_duration
        }


@simulation_behavior_factory_decorator
class RotornetSimulationBehaviorFactory(object):
    """
    Factory to produce RotorNet simulation behavior
    """

    @classmethod
    def generate_behavior(cls, behavior_configuration, simulation_configuration, traffic_generator):
        return RotornetSimulationBehavior(
            traffic_generator=traffic_generator,
            day_duration=behavior_configuration.day_duration,
            night_duration=behavior_configuration.night_duration
        )


class RotornetSimulationBehaviorSparseArrivals(RotornetSimulationBehavior):
    """
    Class implementing a simulation behavior for rotornet topology simulations. This behavior is similar to
    SimpleSimulationBehavior but inserts a RotorReconfigEvent at TimeStamp 1.
    """

    def __init__(self, traffic_generator, day_duration, night_duration=constants.ROTORSWITCH_NIGHT_DURATION):
        super(RotornetSimulationBehaviorSparseArrivals, self).__init__(
            traffic_generator, day_duration, night_duration
        )
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def _create_first_events(self, simulator):
        first_flow = self.traffic_generator.__next__()
        simulator.add_event(
            event.ArrivalEvent(first_flow.arrival_time, first_flow)
        )
        self.logger.info(f"Added initial arrival event.")

        # First event after one period
        simulator.add_event(event.RotorReconfigEvent(event_time=0, day_duration=self.day_duration,
                                                     do_disconnect=False, night_duration=self.night_duration))
        simulator.sort_event_list()

        self.logger.info("Simulator started: {}".format(simulator))


class RotornetSimulationBehaviorSparseArrivalsConfiguration(SimulationBehaviorConfiguration):
    """
    Configuration of Rotornet Simulation Behavior
    """

    def __init__(self, day_duration=constants.ROTORSWITCH_RECONF_PERIOD,
                 night_duration=constants.ROTORSWITCH_NIGHT_DURATION):
        """
        Construct object.
        """
        super(RotornetSimulationBehaviorSparseArrivalsConfiguration, self).__init__(
            factory=RotornetSimulationBehaviorSparseArrivalsFactory.__name__
        )
        self.day_duration = day_duration
        self.night_duration = night_duration

    @property
    def params(self):
        return {
            'day_duration': self.day_duration,
            'night_duration': self.night_duration
        }


@simulation_behavior_factory_decorator
class RotornetSimulationBehaviorSparseArrivalsFactory(object):
    """
    Factory to produce RotorNet simulation behavior
    """

    @classmethod
    def generate_behavior(cls, behavior_configuration, simulation_configuration, traffic_generator):
        return RotornetSimulationBehaviorSparseArrivals(
            traffic_generator=traffic_generator,
            day_duration=behavior_configuration.day_duration,
            night_duration=behavior_configuration.night_duration
        )
