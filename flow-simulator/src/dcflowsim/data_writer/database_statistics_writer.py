from abc import abstractmethod

from . import connection_manager, factory, peewee_interface
from dcflowsim.statistic_collector.flow_statistic_collectors import SimpleFlowCompletionTimeCollector
from .statistic_listener import StatisticListener, StatisticListenerConfiguration
from dcflowsim.utils.factory_decorator import factory_decorator
from .statistic_listener_factory import STATISTIC_LISTENER_FACTORY


class AbstractDatabaseStatisticsWriter(StatisticListener):
    """
    Abstract writer for statistics to database

    """

    def __init__(self, connection_manager, simulation_id):
        super(AbstractDatabaseStatisticsWriter, self).__init__()
        self.connection_manager = connection_manager
        self.simulation_id = simulation_id
        self.next_index_to_write = 0

    @abstractmethod
    def notify_write(self, statistics_collector):
        """
        Writes all new data since last write to database
        Args:
            statistics_collector:

        Returns:

        """
        raise NotImplementedError

    def reset_index(self):
        self.next_index_to_write = 0


class AbstractDatabaseStatisticsWriterFactory(object):
    def get_simulation_id(self, simulation_builder, connection_manager):
        # Add a simulation entry - hardcoded
        flowgen_id = factory.produce_interface(
            simulation_builder.flow_generation_configuration, connection_manager).save(
            simulation_builder.flow_generation_configuration)
        algo_id = factory.produce_interface(
            simulation_builder.algorithm_configuration,
            connection_manager).save(
            simulation_builder.algorithm_configuration)
        topology_id = factory.produce_interface(
            simulation_builder.topology_configuration,
            connection_manager).save(
            simulation_builder.topology_configuration)

        return peewee_interface.SimulationInterface(connection_manager.connection).create_or_get_simulation_id(
            topology_id=topology_id,
            flowgen_id=flowgen_id,
            algo_id=algo_id,
            simulation_duration=simulation_builder.simulation_configuration.simulation_time,
            sim_behavior_config=simulation_builder.simulation_configuration.behavior_configuration,
            sim_builder_name=simulation_builder.simulation_configuration.factory.__name__
        )

    @abstractmethod
    def produce(self, writer_configuration, simulation_builder):
        raise NotImplementedError


class FlowCompletionTimeDatabaseWriter(AbstractDatabaseStatisticsWriter):
    """
    Concrete writer for FCT to database

    """

    def __init__(self, connection_manager, simulation_id):
        super(FlowCompletionTimeDatabaseWriter, self).__init__(connection_manager, simulation_id)
        # Create the concrete peewee interface
        self.interface = peewee_interface.FlowCompletionTimeInterface(connection_manager.connection)

    def notify_write(self, statistics_collector):
        """
        Writes all new flow completion times since last write to database
        Args:
            statistics_collector:
        """
        assert isinstance(statistics_collector, SimpleFlowCompletionTimeCollector)

        data_to_write = statistics_collector.raw_data[self.next_index_to_write:]
        self.interface.save_many(
            data_to_write,
            self.simulation_id
        )
        self.next_index_to_write = len(statistics_collector.raw_data)


@factory_decorator(factory_dict=STATISTIC_LISTENER_FACTORY)
class FlowCompletionTimeDatabaseWriterFactory(AbstractDatabaseStatisticsWriterFactory):

    def produce(self, writer_configuration, simulation_builder):
        """
        Produces the FlowCompletionTimeDatabaseWriter

        Args:
            writer_configuration: containing the db_config and the simulation_id
            simulation_builder:

        Returns:
            FlowCompletionTimeDatabaseWriter
        """

        con_man = connection_manager.DatabaseConnectionManager(
            writer_configuration.db_config
        )
        simulation_id = self.get_simulation_id(
            simulation_builder=simulation_builder,
            connection_manager=con_man
        )

        return FlowCompletionTimeDatabaseWriter(
            connection_manager=con_man,
            simulation_id=simulation_id
        )


class FlowCompletionTimeDatabaseWriterConfiguration(StatisticListenerConfiguration):

    def __init__(self, db_config):
        super(FlowCompletionTimeDatabaseWriterConfiguration, self).__init__(
            factory=FlowCompletionTimeDatabaseWriterFactory.__name__,
            metric=SimpleFlowCompletionTimeCollector.__name__
        )
        self.__db_config = db_config

    @property
    def db_config(self):
        return self.__db_config

    @property
    def params(self):
        return self.__dict__
