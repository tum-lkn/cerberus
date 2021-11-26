import logging
from abc import ABC, abstractmethod

import peewee
import simplejson as json

from . import database_model as dbm
from dcflowsim import constants


def connection(func):
    """
    Decorator to ensure that Database connection is always open
    Args:
        func:

    Returns:

    """

    def con_wrapped(self, *args, **kwargs):
        if self.connection.is_closed():
            with self.connection:
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return con_wrapped


class AbstractPeeweeInterface(ABC):
    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @abstractmethod
    def get_id(self, entity):
        """
        Retrieves the database id (primary key) of the entity
        Args:
            entity:

        Returns:
            ID in database
        """
        raise NotImplementedError

    def exists(self, entity):
        """
        Checks if entity exists in database
        Args:
            entity:

        Returns:
            True if entity exists else false
        """
        try:
            self.get_id(entity)
            return True
        except peewee.DoesNotExist:
            return False

    @abstractmethod
    def get(self, entity_id):
        """
        Retrieves the values for the given entity id and creates the corresponding object
        Args:
            entity_id:

        Returns:
            entity
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, entity):
        """
        Saves entity to the database
        Args:
            entity:

        Returns:
            the database id of the entity
        """
        raise NotImplementedError


class ArrivalTimeGeneratorInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(ArrivalTimeGeneratorInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        return dbm.ArrivalTimeGenerator.get(
            (dbm.ArrivalTimeGenerator.name == entity.__class__.__name__) &
            (dbm.ArrivalTimeGenerator.parameter == json.dumps(entity.params))
        ).atgen_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)
        db_entry = dbm.ArrivalTimeGenerator.create(
            name=entity.__class__.__name__,
            parameter=json.dumps(entity.params)
        )
        return db_entry.atgen_id


class FlowVolumeGeneratorInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(FlowVolumeGeneratorInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        return dbm.FlowVolumeGenerator.get(
            (dbm.FlowVolumeGenerator.name == entity.__class__.__name__) &
            (dbm.FlowVolumeGenerator.parameter == json.dumps(entity.params))
        ).fvgen_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)
        db_entry = dbm.FlowVolumeGenerator.create(
            name=entity.__class__.__name__,
            parameter=json.dumps(entity.params)
        )
        return db_entry.fvgen_id


class ConnectionPairGeneratorInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(ConnectionPairGeneratorInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        return dbm.ConnectionPairGenerator.get(
            (dbm.ConnectionPairGenerator.name == entity.__class__.__name__) &
            (dbm.ConnectionPairGenerator.parameter == json.dumps(entity.params))
        ).cpgen_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)
        db_entry = dbm.ConnectionPairGenerator.create(
            name=entity.__class__.__name__,
            parameter=json.dumps(entity.params)
        )
        return db_entry.cpgen_id


class RandomWithConnectionProbabilityFlowGeneratorInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(RandomWithConnectionProbabilityFlowGeneratorInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        atgen_id = ArrivalTimeGeneratorInterface(self.connection).get_id(
            entity.arrival_time_generation_configuration
        )
        fvgen_id = FlowVolumeGeneratorInterface(self.connection).get_id(
            entity.volume_generation_configuration
        )
        cpgen_id = ConnectionPairGeneratorInterface(self.connection).get_id(
            entity.connection_generation_configuration
        )

        return dbm.FlowGenerator.get(
            (dbm.FlowGenerator.name == entity.__class__.__name__) &
            (dbm.FlowGenerator.seed == entity.seed) &
            (dbm.FlowGenerator.number_flows == entity.max_num_flows) &
            (dbm.FlowGenerator.fk_arrival_time_generator == atgen_id) &
            (dbm.FlowGenerator.fk_flow_volume_generator == fvgen_id) &
            (dbm.FlowGenerator.fk_connection_pair_generator == cpgen_id)
        ).flowgen_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)

        atgen_id = ArrivalTimeGeneratorInterface(self.connection).save(
            entity.arrival_time_generation_configuration
        )
        fvgen_id = FlowVolumeGeneratorInterface(self.connection).save(
            entity.volume_generation_configuration
        )
        cpgen_id = ConnectionPairGeneratorInterface(self.connection).save(
            entity.connection_generation_configuration
        )

        db_entry = dbm.FlowGenerator.create(
            name=entity.__class__.__name__,
            seed=entity.seed,
            number_flows=entity.max_num_flows,
            fk_arrival_time_generator=atgen_id,
            fk_flow_volume_generator=fvgen_id,
            fk_connection_pair_generator=cpgen_id
        )
        return db_entry.flowgen_id


class AlgorithmInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(AlgorithmInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        return dbm.Algorithm.get(
            (dbm.Algorithm.name == entity.__class__.__name__) &
            (dbm.Algorithm.parameter == json.dumps(entity.params))
        ).algorithm_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)
        db_entry = dbm.Algorithm.create(
            name=entity.__class__.__name__,
            parameter=json.dumps(entity.params)
        )
        return db_entry.algorithm_id


class TopologyInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(TopologyInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        decorated_topology_id = None
        if entity.decorated_topology_configuration is not None:
            decorated_topology_id = self.get_id(entity.decorated_topology_configuration)
        return dbm.Topology.get(
            (dbm.Topology.name == entity.__class__.__name__) &
            (dbm.Topology.parameter == json.dumps(entity.params)) &
            (dbm.Topology.fk_decorated_topology == decorated_topology_id)
        ).topology_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        if self.exists(entity):
            return self.get_id(entity)
        decorated_topology_id = None
        if entity.decorated_topology_configuration is not None:
            decorated_topology_id = self.save(entity.decorated_topology_configuration)
        db_entry = dbm.Topology.create(
            name=entity.__class__.__name__,
            parameter=json.dumps(entity.params),
            fk_decorated_topology=decorated_topology_id
        )
        return db_entry.topology_id


class SimulationInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(SimulationInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        raise NotImplementedError

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        raise NotImplementedError

    @connection
    def create_or_get_simulation_id(self, flowgen_id, algo_id, topology_id, simulation_duration=None,
                                    sim_behavior_config=None,
                                    sim_builder_name=None):
        """
        Return simulation id for the
        Args:
            flowgen_id:
            algo_id:
            topology_id:
            simulation_duration: Duration of simulation
            sim_behavior_config: class name of simulation behavior
            sim_builder_name: class name of simulation builder

        Returns:
            simulation id
        """
        entry, created = dbm.Simulation.get_or_create(
            fk_topology=topology_id,
            fk_flow_generator=flowgen_id,
            fk_algorithm=algo_id,
            duration=simulation_duration,
            simulation_behavior_name=sim_behavior_config.factory,
            simulation_behavior_params=sim_behavior_config.params,
            simulation_builder_name=sim_builder_name
        )
        return entry.simulation_id

    @connection
    def is_finished(self, simulation_id):
        """
        Check if finished flag is set for the given simulation id
        Args:
            simulation_id: database id of simulation

        Returns:
            The value of the finished (true if already done)
        """
        return dbm.Simulation.get(dbm.Simulation.simulation_id == simulation_id).finished

    @connection
    def set_finished(self, simulation_id):
        dbm.Simulation.update({dbm.Simulation.finished: True}).where(
            dbm.Simulation.simulation_id == simulation_id).execute()

    @connection
    def insert_start_time(self, simulation_id, t_start):
        dbm.Simulation.update({dbm.Simulation.start_time: t_start}).where(
            dbm.Simulation.simulation_id == simulation_id).execute()

    @connection
    def insert_end_time(self, simulation_id, t_end):
        dbm.Simulation.update({dbm.Simulation.end_time: t_end}).where(
            dbm.Simulation.simulation_id == simulation_id).execute()

    @connection
    def insert_commit_sha(self, simulation_id, sha):
        dbm.Simulation.update({dbm.Simulation.sha: sha}).where(
            dbm.Simulation.simulation_id == simulation_id).execute()


class FlowInterface(AbstractPeeweeInterface):
    class __FlowIdCache(object):
        """
        Provides a mapping between global_ids of flows and the corresponding id in the database
        """

        def __init__(self):
            self.__cache = dict()

        def get_cached_id(self, flow_global_id):
            try:
                return self.__cache[flow_global_id]
            except KeyError:
                return None

        def add_id_to_cache(self, flow_global_id, database_id):
            self.__cache[flow_global_id] = database_id

    def __init__(self, connection):
        super(FlowInterface, self).__init__(connection)
        self.cache = FlowInterface.__FlowIdCache()

    @connection
    def get_id(self, entity):
        # First try to retrieve id from cache
        database_id = self.cache.get_cached_id(entity.global_id)
        if not database_id:
            # Id not cached yet. Retrieve from database
            database_id = dbm.Flow.get(
                (dbm.Flow.source_node_id == entity.source.node_id) &
                (dbm.Flow.destination_node_id == entity.destination.node_id) &
                (dbm.Flow.volume == entity.volume) &
                (dbm.Flow.arrival_time == entity.arrival_time) &
                (dbm.Flow.flow_global_id == entity.global_id)
            ).flow_id

            # Add id to cache
            self.cache.add_id_to_cache(entity.global_id, database_id)
        return database_id

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        try:
            db_id = dbm.Flow.create(
                flow_global_id=entity.global_id,
                source_node_id=entity.source.node_id,
                destination_node_id=entity.destination.node_id,
                volume=entity.volume,
                arrival_time=entity.arrival_time
            ).flow_id
        except peewee.IntegrityError:
            db_id = self.get_id(entity)
        return db_id


class FlowCompletionTimeInterface(AbstractPeeweeInterface):
    def __init__(self, connection):
        super(FlowCompletionTimeInterface, self).__init__(connection)

    @connection
    def get_id(self, entity):
        raise NotImplementedError

    @connection
    def get(self, entity_id):
        raise NotImplementedError

    @connection
    def save(self, entity):
        raise NotImplementedError

    @connection
    def save_many(self, results, simulation_id):
        """
        Saves all completion times in results
        Args:
            results: list containing FlowCompletionTimeResults objects
            simulation_id: database id of simulation run

        Returns:
            the number of inserted rows
        """
        if len(results) == 0:
            self.logger.info("No elements to insert.")
            return 0
        flow_interface = FlowInterface(self.connection)
        inserts = list()
        rows_inserted = 0
        self.logger.info("{} elements to insert.".format(len(results)))
        for res in results:
            inserts.append(
                {
                    'fk_simulation': simulation_id,
                    'fk_flow': flow_interface.save(res.flow),
                    'completion_time': res.value,
                    'first_time_with_rate': res.first_time_with_rate
                }
            )

            if len(inserts) > constants.PEEWEE_INTERFACE_INSERT_MANY_SIZE:
                rows_inserted += dbm.FlowCompletionTime.insert_many(
                    inserts
                ).execute()
                inserts = list()
        if len(inserts) > 0:
            rows_inserted += dbm.FlowCompletionTime.insert_many(
                inserts
            ).execute()
        self.logger.info("Inserted {} rows.".format(rows_inserted))
        return rows_inserted
