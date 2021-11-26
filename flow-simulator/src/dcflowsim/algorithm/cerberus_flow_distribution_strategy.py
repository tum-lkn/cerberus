import logging

from dcflowsim import configuration
from dcflowsim.utils.number_conversion import convert_to_decimal
from dcflowsim.algorithm.score_based_algorithm import GreedyOpticalScoreBasedAlgorithm


class AbstractFlowDistributionStrategy(object):
    def initialize_strategy(self, flows, algorithm_object):
        raise NotImplementedError

    def notify_add_flow_strategy(self, new_flow, algorithm_object):
        raise NotImplementedError

    def notify_remove_flow_strategy(self, removed_flow, algorithm_object):
        raise NotImplementedError

    def notify_partial_rem_flow(self, removed_flow, algorithm_object):
        raise NotImplementedError


class FixedThresholdsDynamicFlowDistributionStrategy(AbstractFlowDistributionStrategy):
    """
    Distribution behavior with fixed thresholds. If flows cannot be added to DA, they are allocated to RotorNet
    """

    def __init__(self, medium_threshold, large_threshold):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.medium_threshold = convert_to_decimal(medium_threshold)
        self.large_threshold = convert_to_decimal(large_threshold)

        self.__flows_moved_from_da_to_rotor = {}  # Key: flow_id, Value: flow object

        self.__da_flow_ids = set()
        self.__rotor_flow_ids = set()

    def initialize_strategy(self, flows, algorithm_object):
        """
        Classifies flows by their size and forwards them to initialization function of corresponding algorithm,
        DA flows are assigned to rotornet if they do not fit
        Args:
            flows: list of all flows already in environment
            algorithm_object: Cerberus algorithm object

        Returns:

        """
        # Check if DA algo is properly configured
        assert algorithm_object.da_algo.raise_no_allocation
        large_flows = list()
        medium_flows = list()
        small_flows = list()
        # Classify flows
        for flow_obj in flows:
            if flow_obj.volume >= self.large_threshold:
                large_flows.append(flow_obj)
                algorithm_object.flows_to_algo[flow_obj.global_id] = algorithm_object.da_ev_listener
                # Keep additionally track of DA flow ids
                self.__da_flow_ids.add(flow_obj.global_id)
            elif flow_obj.volume >= self.medium_threshold:
                medium_flows.append(flow_obj)
                algorithm_object.flows_to_algo[flow_obj.global_id] = algorithm_object.rotor_ev_listener
                # Keep additionally track of rotor flow ids
                self.__rotor_flow_ids.add(flow_obj.global_id)
            else:
                small_flows.append(flow_obj)
                algorithm_object.flows_to_algo[flow_obj.global_id] = algorithm_object.static_ev_listener

        # Initialize ev listeners with their corresponding flows.
        for ev in algorithm_object.da_ev_listener:
            ev.initialize(large_flows)
        # Take not preallocated flows and assign to rotor
        # We assume we have a GreedyOpticalScoreBased algorithm
        for flow_obj in algorithm_object.da_algo.not_allocated_flows:
            self.logger.debug(f"Assigning large flow {flow_obj} to Rotor.")
            medium_flows.append(flow_obj)
            self.__flows_moved_from_da_to_rotor[flow_obj.global_id] = flow_obj
            self.__da_flow_ids.remove(flow_obj.global_id)
            self.__rotor_flow_ids.add(flow_obj.global_id)
            algorithm_object.flows_to_algo[flow_obj.global_id] = algorithm_object.rotor_ev_listener
        for ev in algorithm_object.rotor_ev_listener:
            ev.initialize(medium_flows)
        for ev in algorithm_object.static_ev_listener:
            ev.initialize(small_flows)

    def notify_add_flow_strategy(self, new_flow, algorithm_object):
        """
        Handle flow arrivals. Check for the flow size and allocate to corresponding topology based on size thresholds
        Args:
            new_flow:
            algorithm_object: Instance of Cerberus algorithm

        Returns:

        """
        # Check if DA algo is properly configured
        assert algorithm_object.da_algo.raise_no_allocation
        # Check the size of the flow
        if new_flow.volume >= self.large_threshold:
            self.logger.debug(
                f"Flow {new_flow.global_id}: {new_flow.volume} >= {self.large_threshold} - assigning to DA")
            # Notify DA algo and ev listeners. IMPORTANT notify algo last so that flow list etc. are updated before
            try:
                for ev in algorithm_object.da_ev_listener:
                    ev.notify_add_flow(new_flow)
                algorithm_object.flows_to_algo[new_flow.global_id] = algorithm_object.da_ev_listener
                self.__da_flow_ids.add(new_flow.global_id)
            except GreedyOpticalScoreBasedAlgorithm.FlowAllocationError:
                self.logger.debug(f"Flow could not be allocated to DA")
                # Notify Rotor algo and ev listeners.
                for ev in algorithm_object.rotor_ev_listener:
                    ev.notify_add_flow(new_flow)
                self.__flows_moved_from_da_to_rotor[new_flow.global_id] = new_flow
                algorithm_object.flows_to_algo[new_flow.global_id] = algorithm_object.rotor_ev_listener
                self.__rotor_flow_ids.add(new_flow.global_id)
                # We keep track of DA flows that could not be allocated in da_algo.not_allocated_flows...
                # maybe we're lucky later on
        elif new_flow.volume >= self.medium_threshold:
            self.logger.debug(
                f"Flow {new_flow.global_id}: {new_flow.volume} >= {self.medium_threshold} - assigning to Rotor")
            # Notify Rotor algo and ev listeners. IMPORTANT notify algo last so that flow list etc. are updated before
            for ev in algorithm_object.rotor_ev_listener:
                ev.notify_add_flow(new_flow)
            algorithm_object.flows_to_algo[new_flow.global_id] = algorithm_object.rotor_ev_listener
        else:
            self.logger.debug(
                f"Flow {new_flow.global_id}: {new_flow.volume} < {self.medium_threshold} - assigning to Static")
            # Notify static algo and ev listeners. IMPORTANT notify algo last so that flow list etc. are updated before
            for ev in algorithm_object.static_ev_listener:
                ev.notify_add_flow(new_flow)
            algorithm_object.flows_to_algo[new_flow.global_id] = algorithm_object.static_ev_listener

    def notify_remove_flow_strategy(self, removed_flow, algorithm_object):
        # Check if DA algo is properly configured
        assert algorithm_object.da_algo.raise_no_allocation
        # If this flow was a DA one, DA algo might allocate another flow from the pending ones. We do not care about
        # RotorSlots for now since the partial flows make sure that we do not route any bit twice.
        promoted_flows = list()
        for ev in algorithm_object.flows_to_algo[removed_flow.global_id]:
            res = ev.notify_rem_flow(removed_flow)
            if res is not None:
                # DA allocated a new flow from the pending ones
                promoted_flows += res
        del algorithm_object.flows_to_algo[removed_flow.global_id]

        if removed_flow.global_id in self.__rotor_flow_ids:
            # If this was a flow moved from DA to Rotor clean it from list of pending flows in DA algo and return
            self.__rotor_flow_ids.remove(removed_flow.global_id)
            if removed_flow.global_id in self.__flows_moved_from_da_to_rotor:
                algorithm_object.da_algo.remove_pending_flow(removed_flow)
                del self.__flows_moved_from_da_to_rotor[removed_flow.global_id]
                return

        elif removed_flow.global_id in self.__da_flow_ids:
            self.logger.debug(f"DA allocated {len(promoted_flows)} flows from Rotor.")
            for promoted_flow_id in promoted_flows:
                # Update assignment
                algorithm_object.flows_to_algo[promoted_flow_id] = algorithm_object.da_ev_listener
                del self.__flows_moved_from_da_to_rotor[promoted_flow_id]
                self.__rotor_flow_ids.remove(promoted_flow_id)
                self.__da_flow_ids.add(promoted_flow_id)

                # Finally, remove from Rotoralgos vision
                promoted_flow = algorithm_object.env.flows[promoted_flow_id]
                for ev in algorithm_object.rotor_ev_listener:
                    ev.notify_rem_flow(promoted_flow)
            self.__da_flow_ids.remove(removed_flow.global_id)

    def notify_partial_rem_flow(self, removed_flow, algorithm_object):
        for ev in algorithm_object.rotor_ev_listener + algorithm_object.da_ev_listener + algorithm_object.static_ev_listener:
            ev.notify_partial_rem_flow(removed_flow)


class FixedThresholdsDynamicFlowDistributionStrategyFactory(object):
    @classmethod
    def generate_behavior(cls, behavior_configuration):
        return FixedThresholdsDynamicFlowDistributionStrategy(
            medium_threshold=behavior_configuration.medium_threshold,
            large_threshold=behavior_configuration.large_threshold
        )


class FixedThresholdsDynamicFlowDistributionStrategyConfiguration(configuration.AbstractConfiguration):
    def __init__(self, medium_threshold, large_threshold):
        super(FixedThresholdsDynamicFlowDistributionStrategyConfiguration, self).__init__(
            factory=FixedThresholdsDynamicFlowDistributionStrategyFactory.__name__
        )
        self.medium_threshold = convert_to_decimal(medium_threshold)
        self.large_threshold = convert_to_decimal(large_threshold)

    @property
    def params(self):
        return {
            'name': self.__class__.__name__,
            'medium_threshold': self.medium_threshold,
            'large_threshold': self.large_threshold
        }


FLOW_DISTRIBUTION_STRATEGY_FACTORIES = dict()
FLOW_DISTRIBUTION_STRATEGY_FACTORIES[
    FixedThresholdsDynamicFlowDistributionStrategyFactory.__name__
] = FixedThresholdsDynamicFlowDistributionStrategyFactory


def produce_flow_distribution_strategy(strategy_configuration):
    return FLOW_DISTRIBUTION_STRATEGY_FACTORIES[strategy_configuration.factory]().generate_behavior(
        strategy_configuration
    )
