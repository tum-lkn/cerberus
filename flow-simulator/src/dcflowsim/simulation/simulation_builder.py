import abc
import logging

from dcflowsim.network import topology_factories
from dcflowsim.algorithm import algorithm_factory
from dcflowsim.flow_generation import flow_generator_factories
from dcflowsim.data_writer import statistic_listener_factory
from dcflowsim.statistic_collector import statistic_collector_factory

from ..environment import environment, event_inserter, flow
from . import simulation, simulation_behavior, event


class AbstractSimulationBuilder(abc.ABC):

    @abc.abstractmethod
    def build_simulation(self):
        raise NotImplementedError


class SimpleSimulationBuilder(AbstractSimulationBuilder):

    def __init__(self, simulation_configuration):
        """
        Init method

        Args:
            simulation_configuration: contains all configurations.
        """
        self.simulation_configuration = simulation_configuration
        self.topology_configuration = simulation_configuration.topology_configuration
        self.behavior_configuration = simulation_configuration.behavior_configuration
        self.algorithm_configuration = simulation_configuration.algorithm_configuration
        self.flow_generation_configuration = simulation_configuration.flow_generation_configuration
        self.statistics_configuration = simulation_configuration.statistics_configuration
        self.data_model_configuration = simulation_configuration.data_model_configuration
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def build_simulation(self):
        """
        Builds and returns a simulation object ready to be executed.

        Returns:

        """

        topology = self._build_topology()
        env = environment.Environment(topology)
        simulator = self._build_simulator(env)
        self._build_algorithm(env)
        flow_gen = self._build_traffic_generator(env)

        statistic_collectors = self._build_statistics(env)
        self._build_data_collectors(statistic_collectors)

        simulator.behavior = self._build_simulation_behavior(flow_gen)
        self._build_events_and_init_environment(simulator, flow_gen)

        return simulator

    def _build_simulation_behavior(self, traffic_generator):
        """
        Build the simulation behavior.

        Returns:
            simulation behavior
        """
        return simulation_behavior.produce(behavior_configuration=self.behavior_configuration,
                                           simulation_configuration=self.simulation_configuration,
                                           traffic_generator=traffic_generator)

    def _build_statistics(self, env):
        """
        Build statistics collector. There are potentially many collectors.

        Returns:
            list of statistic collectors
        """
        statistic_collectors = list()

        for configuration in self.statistics_configuration:
            statistic_collectors.append(
                statistic_collector_factory.produce(env, configuration)
            )

        for collector in statistic_collectors:
            env.add_environment_listener(collector)

        return statistic_collectors

    def _build_topology(self):
        """
        Build topology.

        Returns:
            topology
        """
        self.logger.info("Building topology.")

        return topology_factories.TopologyBuilder.build_topologies(self.topology_configuration)

    def _build_simulator(self, env):
        self.logger.info("Creating simulator")
        return simulation.Simulator(env, simulation_time=self.simulation_configuration.simulation_time)

    def _build_algorithm(self, env):
        """
        Build algorithm.

        Returns:
            algorithm
        """
        self.logger.info("Building algorithm: {}".format(
            self.algorithm_configuration.factory
        ))

        alg = algorithm_factory.produce(self.algorithm_configuration, env)
        env.add_environment_listener(alg)

        return alg

    def _build_traffic_generator(self, env):
        """
        Build traffic generator.

        Returns:
            traffic generator.
        """
        self.logger.info("Building traffic generator.")

        return flow_generator_factories.produce_flow_generator(
            self.flow_generation_configuration, env)

    def _build_data_collectors(self, statistic_collectors):
        """
        Build data collectors and attaches them to the corresponding StatisticListeners

        Returns:
            data_collectors
        """

        self.logger.info("Building data model.")

        data_collectors = []

        for config in self.data_model_configuration:
            this_collector = statistic_listener_factory.produce(config, self)
            stat_collector = self.__get_statistic_collector_by_type(
                statistic_collectors,
                config.metric
            )
            stat_collector.add_statistics_listener(this_collector)
            data_collectors.append(
                this_collector
            )

        return data_collectors

    def _build_events_and_init_environment(self, simulator, flowgenerator):
        """
        Creates the initial state of the environment and intializes the EventListeners. Adds events to the event list
        if necessary to represent state of environment.

        Returns:

        """
        self.logger.info("Initialize environment")
        for envlist in list(simulator.environment.listeners):
            envlist.initialize(simulator.environment.flows.values())

    def __get_statistic_collector_by_type(self, statistic_collectors, requested_type):
        """
        Iterates over the list of statistic_collectors and returns the first instance that has the requested type
        Args:
            statistic_collectors: list of StatisticCollector instances
            requested_type: requested concrete statistic collector, i.e., metric

        Returns:

        """
        for stat_col in statistic_collectors:
            if stat_col.__class__.__name__ == requested_type:
                return stat_col
        self.logger.error("Requested statistic collector could not be found.")
        raise RuntimeError("Requested statistic collector could not be found.")


class InitialEnvStateAllFlowsSimulationBuilder(SimpleSimulationBuilder):
    """
    Assumes that all flows arrive at zero. Therefore, it adds all but one to the environment before starting the sim.
    Adds also corresponding ServiceEvents. The final flow is added with an arrival event.
    """

    def _build_events_and_init_environment(self, simulator, flowgenerator):
        """
        Creates the initial state of the environment and intializes the EventListeners. Adds all but one flow to the
        environment and calls initialization. Adds required service events and an ArrivalEvent for the last flow.
        Flowgenerator has been consumed then, so SimBehavior iterating over it does not have any effect.

        Returns:

        """
        flow_container = dict()
        # service_events_to_add = list()
        f = None
        for f in flowgenerator:
            flow_container[f.global_id] = f
            f.add_listener(
                flow.ServiceEventCreator(simulator)
            )
            f.add_listener(simulator.environment)

        if f:
            # Delete last flow from env since we want to have an Arrival for it
            del flow_container[f.global_id]
            f.reset_listeners()
            simulator.add_event(
                event.ArrivalEvent(f.arrival_time, f)
            )
        # Update environment and initialize listeners
        simulator.environment.set_flows(flow_container)
        for envlist in list(simulator.environment.listeners):
            envlist.initialize(simulator.environment.flows.values())

        simulator.sort_event_list()


class SimpleSimulationBuilderWithEventInsertion(SimpleSimulationBuilder):
    """
    Simple simulation builder with the ability to insert events during simulation
    """

    def _build_simulator(self, env):
        sim = super(SimpleSimulationBuilderWithEventInsertion, self)._build_simulator(env)
        sim.environment.add_environment_listener(event_inserter.EventInserter(sim.environment, sim))
        return sim


class InitialEnvStateAllFlowsSimulationWithEventInsertionBuilder(InitialEnvStateAllFlowsSimulationBuilder):
    """
    Combination of InitialEnvStateAllFlowsSimulation and EventInsertion builders. Follows mainly
    InitialEnvStateAllFlowsSimulation and adds EventInserter in the end as additional EnvListener
    """

    def _build_simulator(self, env):
        sim = super(InitialEnvStateAllFlowsSimulationWithEventInsertionBuilder, self)._build_simulator(env)
        sim.environment.add_environment_listener(event_inserter.EventInserter(sim.environment, sim))
        return sim
