import logging

from .cerberus_flow_distribution_strategy import produce_flow_distribution_strategy
from .abstract_algorithm import AbstractAlgorithmConfiguration, AbstractAlgorithmFactory
from .algorithm_factory import algorithm_factory_decorator

from dcflowsim import constants
from dcflowsim.environment import environment_listener
from dcflowsim.network import ocs_topology
from dcflowsim.data_writer.factory import peewee_interface_factory_decorator
from dcflowsim.data_writer.peewee_interface import AlgorithmInterface


class CerberusAlgorithm(environment_listener.EnvironmentListener):
    def __init__(self, env, static_algo, rotor_algo, da_algo, static_ev_listener, rotor_ev_listener, da_ev_listener,
                 flow_distribution_strategy):
        super(CerberusAlgorithm, self).__init__(constants.ALGORITHM_PRIORITY)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = env
        self.static_algo = static_algo
        assert isinstance(self.static_algo, environment_listener.EnvironmentListener)
        self.rotor_algo = rotor_algo
        assert isinstance(self.rotor_algo, environment_listener.EnvironmentListener)
        self.da_algo = da_algo
        assert isinstance(self.da_algo, environment_listener.EnvironmentListener)

        self.static_ev_listener = static_ev_listener + [self.static_algo]
        self.rotor_ev_listener = rotor_ev_listener + [self.rotor_algo]
        self.da_ev_listener = da_ev_listener + [self.da_algo]

        self.flow_distribution_strategy = flow_distribution_strategy

        # Keep track of where flows are allocated
        self.flows_to_algo = dict()  # Map flow id to algo

    def initialize(self, flows):
        self.flow_distribution_strategy.initialize_strategy(flows, self)

    def get_sub_listeners(self):
        return list()

    def notify_add_flow(self, new_flow):
        self.flow_distribution_strategy.notify_add_flow_strategy(new_flow, self)

    def notify_rem_flow(self, removed_flow):
        self.flow_distribution_strategy.notify_remove_flow_strategy(removed_flow, self)

    def notify_partial_rem_flow(self, removed_flow):
        self.flow_distribution_strategy.notify_partial_rem_flow(removed_flow, self)

    def notify_upd_time(self):
        # Notify all environment listeners / forward to everything
        for ev in self.rotor_ev_listener + self.da_ev_listener + self.static_ev_listener:
            ev.notify_upd_time()

    def notify_upd_ocs(self, ocs):
        # Notify all environment listeners / forward to everything
        for ev in self.rotor_ev_listener + self.da_ev_listener + self.static_ev_listener:
            ev.notify_upd_ocs(ocs)

    def notify_upd_req_ocs(self, ocs, circuits):
        # Notify all environment listeners / forward to everything
        for ev in self.rotor_ev_listener + self.da_ev_listener + self.static_ev_listener:
            ev.notify_upd_req_ocs(ocs, circuits)

    def notify_upd_flow_route(self, changed_flow):
        # Notify all environment listeners / forward to everything
        for ev in self.rotor_ev_listener + self.da_ev_listener + self.static_ev_listener:
            ev.notify_upd_flow_route(changed_flow)


@algorithm_factory_decorator
class CerberusAlgorithmFactory(AbstractAlgorithmFactory):
    @classmethod
    def generate_algorithm(cls, algorithm_configuration, env):
        from .algorithm_factory import produce
        static_algo = produce(algorithm_configuration.static_algorithm, env)
        # Get sub-EnvironmentListeners
        static_ev_listener = static_algo.get_sub_listeners()

        rotor_algo = produce(algorithm_configuration.rotor_algorithm, env)
        rotor_ev_listener = rotor_algo.get_sub_listeners()

        da_algo = produce(algorithm_configuration.da_algorithm, env)
        da_ev_listener = da_algo.get_sub_listeners()

        # De-register sub-EnvironmentListeners. Events are handled by CerberusAlgorithm
        for evlistener in static_ev_listener + rotor_ev_listener + da_ev_listener:
            env.remove_environment_listener(evlistener)

        distribution_strategy = produce_flow_distribution_strategy(
            algorithm_configuration.flow_distribution_strategy_configuration
        )

        return CerberusAlgorithm(
            env=env,
            static_algo=static_algo, rotor_algo=rotor_algo, da_algo=da_algo,
            static_ev_listener=static_ev_listener, rotor_ev_listener=rotor_ev_listener, da_ev_listener=da_ev_listener,
            flow_distribution_strategy=distribution_strategy
        )


@peewee_interface_factory_decorator(interface=AlgorithmInterface)
class CerberusAlgorithmConfiguration(AbstractAlgorithmConfiguration):
    def __init__(self, static_algorithm, rotornet_algorithm, da_algorithm, flow_distribution_strategy_configuration):
        """
        Configuration for Cerberus algorithm. Takes configurations for sub-algorithms. Note that all TopologyViewConfig-
        urations are overwritten.
        Args:
            static_algorithm: Algorithm configuration for algorithm on static topology
            rotornet_algorithm: Algorithm configuration for algorithm on rotor topology
            da_algorithm:  Algorithm configuration for algorithm on demand-aware topology
        """
        super(CerberusAlgorithmConfiguration, self).__init__(
            factory=CerberusAlgorithmFactory.__name__
        )
        self.static_algorithm = static_algorithm
        assert isinstance(self.static_algorithm, AbstractAlgorithmConfiguration)
        self.rotor_algorithm = rotornet_algorithm
        assert isinstance(self.rotor_algorithm, AbstractAlgorithmConfiguration)
        self.da_algorithm = da_algorithm
        assert isinstance(self.da_algorithm, AbstractAlgorithmConfiguration)

        self.flow_distribution_strategy_configuration = flow_distribution_strategy_configuration

        # Set appropriate topology view configurations (keep existing configs)
        if self.static_algorithm.topology_view_configuration is None:
            self.static_algorithm.topology_view_configuration = ocs_topology.TopologyWithOcsViewConfiguration(
                relevant_link_identifiers=[ocs_topology.LinkIdentifiers.static, ocs_topology.LinkIdentifiers.default],
                relevant_ocs_identifiers=[]
            )
        if self.rotor_algorithm.topology_view_configuration is None:
            self.rotor_algorithm.topology_view_configuration = ocs_topology.TopologyWithOcsViewConfiguration(
                relevant_link_identifiers=[ocs_topology.LinkIdentifiers.rotor, ocs_topology.LinkIdentifiers.default],
                relevant_ocs_identifiers=[ocs_topology.OCSSwitchTypes.rotor]
            )
        if self.da_algorithm.topology_view_configuration is None:
            self.da_algorithm.topology_view_configuration = ocs_topology.TopologyWithOcsViewConfiguration(
                relevant_link_identifiers=[ocs_topology.LinkIdentifiers.dynamic, ocs_topology.LinkIdentifiers.default],
                relevant_ocs_identifiers=[ocs_topology.OCSSwitchTypes.ocs]
            )

    @property
    def params(self):
        return {
            'static_algorithm': self.static_algorithm.__class__.__name__,
            'static_algorithm_params': self.static_algorithm.params,
            'rotor_algorithm': self.rotor_algorithm.__class__.__name__,
            'rotor_algorithm_params': self.rotor_algorithm.params,
            'da_algorithm': self.da_algorithm.__class__.__name__,
            'da_algorithm_params': self.da_algorithm.params,
            'flow_distribution_strategy': self.flow_distribution_strategy_configuration.params
        }
