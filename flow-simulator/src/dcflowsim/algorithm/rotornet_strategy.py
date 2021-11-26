from decimal import Decimal

from dcflowsim import configuration
from dcflowsim.utils.number_conversion import convert_to_decimal


class AbstractRotorNetAllocateFlowStrategy(object):
    def __init__(self):
        self._rotornet_algo = None

    def set_rotornet_algorithm(self, rotornet_algo):
        self._rotornet_algo = rotornet_algo

    def allocate(self):
        raise NotImplementedError

    def get_matchings_for_indirect(self) -> list:
        """
        Returns the matchings that shall be considered for allocation of indirect traffic.

        Returns:
            list of tuples (src_tor, dst_tor, rotor_switch)
        """
        raise NotImplementedError


class AbstractRotorNetAllocateFlowStrategyFactory(object):
    def produce(self, config) -> AbstractRotorNetAllocateFlowStrategy:
        raise NotImplementedError


ALLOCATE_INDIRECT_FLOW_STRATEGY_FACTORIES = dict()


def rotornet_strategy_factory_decorator(cls):
    """
    Class decorator for all algorithm factories.

    Args:
        cls: the factory class that should be decorated.

    Returns:
        simply the cls
    """
    ALLOCATE_INDIRECT_FLOW_STRATEGY_FACTORIES[cls.__name__] = cls
    return cls


def produce_rotornet_strategy(
        strategy_config: configuration.AbstractConfiguration) -> AbstractRotorNetAllocateFlowStrategy:
    return ALLOCATE_INDIRECT_FLOW_STRATEGY_FACTORIES[
        strategy_config.factory
    ]().produce(strategy_config)


class AllocateDirectTrafficStrategy(AbstractRotorNetAllocateFlowStrategy):
    def __init__(self):
        super(AllocateDirectTrafficStrategy, self).__init__()
        self._matchings_for_indirect = None

    def allocate(self):
        self._matchings_for_indirect = list()
        # Loop over all rotorswitches and get the connected rack pairs
        for rotor_switch in self._rotornet_algo.rotor_switches:
            # Get the day_duration of the RotorSwitch
            day_duration = rotor_switch.day_duration
            # Max volume to be transmitted per flow
            volume_limit = convert_to_decimal(day_duration * self._rotornet_algo.max_rate_per_flow)

            # First loop to route all of the indirect + direct traffic
            for (torA, torB) in rotor_switch.get_pairs_connected_nodes():
                self._rotornet_algo.initialize_flow_volume_dict_for_tor_pair_route_stored_indirect_traffic(rotor_switch,
                                                                                                           torA, torB)

                # get the direct flows, add the flows to flow list and subtract their volumes from available capacity
                # DIRECT TRAFFIC
                # Get the SuperFlows for the unidirectional demand between the rack pair
                self._rotornet_algo.allocate_direct_traffic(
                    tor_pair=(torA, torB),
                    volume_limit=volume_limit
                )
                assert self._rotornet_algo.volume_dict[(torA, torB)] >= Decimal('0')

                self._matchings_for_indirect.append((torA, torB, rotor_switch))

    def get_matchings_for_indirect(self):
        return self._matchings_for_indirect


@rotornet_strategy_factory_decorator
class AllocateDirectTrafficStrategyFactory(AbstractRotorNetAllocateFlowStrategyFactory):
    def produce(self, config):
        return AllocateDirectTrafficStrategy()


class AllocateDirectTrafficStrategyConfiguration(configuration.AbstractConfiguration):
    def __init__(self):
        super(AllocateDirectTrafficStrategyConfiguration, self).__init__(
            factory=AllocateDirectTrafficStrategyFactory.__name__
        )

    @property
    def params(self):
        return {
            "name": self.factory
        }


class AllToRsAllocateIndirectFlowStrategy(AbstractRotorNetAllocateFlowStrategy):
    def __init__(self, direct_flow_allocation_strategy):
        super(AllToRsAllocateIndirectFlowStrategy, self).__init__()
        self.direct_flow_allocation_strategy = direct_flow_allocation_strategy

    def set_rotornet_algorithm(self, rotornet_algo):
        super(AllToRsAllocateIndirectFlowStrategy, self).set_rotornet_algorithm(rotornet_algo)
        self.direct_flow_allocation_strategy.set_rotornet_algorithm(rotornet_algo)

    def allocate(self):
        self.direct_flow_allocation_strategy.allocate()

        for rotor_switch in self._rotornet_algo.rotor_switches:  # k
            # Get the day_duration of the RotorSwitch
            day_duration = rotor_switch.day_duration
            # Max volume to be transmitted per flow
            volume_limit = convert_to_decimal(day_duration * self._rotornet_algo.max_rate_per_flow)

            # First loop to route all of the indirect + direct traffic
            for (torA, torB) in rotor_switch.get_pairs_connected_nodes():  # n/2
                # Get the node_ids of all ToR switches and remove the currently connected ToRs
                tors_to_check = [
                    tor.node_id for tor
                    in self._rotornet_algo.topology.get_nodes_by_type("TorSwitch")
                ]
                tors_to_check.remove(torA)
                tors_to_check.remove(torB)

                # Check indirect traffic for torA --> torB --> torC
                # Loop over the currently not connected ToRs (torC)
                for torC in tors_to_check:  # n
                    # A --> B --> C
                    self._rotornet_algo.allocate_indirect_flows_one_direction(
                        src_tor_id=torA,
                        intermediate_tor_id=torB,
                        dst_tor_id=torC,
                        tor_key_pair=(torA, torB),
                        rotor_switch=rotor_switch,
                        volume_limit=volume_limit
                    )

                    # B --> A --> C
                    self._rotornet_algo.allocate_indirect_flows_one_direction(
                        src_tor_id=torB,
                        intermediate_tor_id=torA,
                        dst_tor_id=torC,
                        tor_key_pair=(torA, torB),
                        rotor_switch=rotor_switch,
                        volume_limit=volume_limit
                    )

                    assert self._rotornet_algo.volume_dict[(torA, torB)] >= Decimal(
                        '0'), f"{self._rotornet_algo.volume_dict[(torA, torB)]}"

    def get_matchings_for_indirect(self):
        raise RuntimeError


@rotornet_strategy_factory_decorator
class AllToRsAllocateIndirectFlowStrategyFactory(AbstractRotorNetAllocateFlowStrategyFactory):
    def produce(self, config):
        return AllToRsAllocateIndirectFlowStrategy(
            direct_flow_allocation_strategy=produce_rotornet_strategy(config.direct_flow_strategy_configuration)
        )


class AllToRsAllocateIndirectFlowStrategyConfiguration(configuration.AbstractConfiguration):
    def __init__(self, direct_flow_strategy_configuration=None):
        super(AllToRsAllocateIndirectFlowStrategyConfiguration, self).__init__(
            factory=AllToRsAllocateIndirectFlowStrategyFactory.__name__
        )
        self.direct_flow_strategy_configuration = direct_flow_strategy_configuration
        if self.direct_flow_strategy_configuration is None:
            self.direct_flow_strategy_configuration = AllocateDirectTrafficStrategyConfiguration()

    @property
    def params(self):
        return {
            "name": self.factory,
            "direct_flow_strategy_configuration": self.direct_flow_strategy_configuration.params
        }
