import logging
import numpy

from dcflowsim import configuration, constants
from . import environment_listener, flow
from dcflowsim.utils import flow_handling
from decimal import Decimal

DEMAND_MATRIX_FACTORY = dict()


def demand_matrix_factory_decorator(cls):
    """
    Class decorator for all demand matrix factories.

    Args:
        cls: the factory class that should be decorated.

    Returns:
        simply the cls
    """
    DEMAND_MATRIX_FACTORY[cls.__name__] = cls
    return cls


def produce_demand_matrix(config, environment):
    return DEMAND_MATRIX_FACTORY[config.factory]().produce(environment, config)


class AbstractDemandMatrix(environment_listener.EnvironmentListener):
    def __init__(self, environment):
        super(AbstractDemandMatrix, self).__init__(priority=constants.ENVIRONMENT_VIEW_PRIORITY)
        self._environment = environment

    @property
    def environment(self):
        return self._environment

    def get_sub_listeners(self):
        return []


class SimpleTor2TorDemandMatrixWithFlowList(AbstractDemandMatrix, flow.FlowListener):
    """
    Demand matrix for ToR to ToR demand. Should be used with RotorNet topologies.
    Also provides access to all flows that make up the demand between two ToRs.
    """

    def __init__(self, environment):
        super(SimpleTor2TorDemandMatrixWithFlowList, self).__init__(environment)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__matrix = None
        self.__tor_node_id_to_idx_mapping = dict()
        self.__tor_pair_to_flow_mapping = dict()

        self._already_initialized = False

        self.__init_matrix()

    def initialize(self, flows):
        """
        Synchronize initial environment state
        Args:
            flows: list of flows that shall be considered in initial state

        Returns:

        """
        if self._already_initialized:
            return
        self.__tor_pair_to_flow_mapping = dict()
        for flow_obj in flows:
            src_tor_node, dst_tor_node = flow_handling.get_tor_node_ids_of_flow(flow_obj)

            # Map the flow ids to a tor pair
            if (src_tor_node, dst_tor_node) in self.__tor_pair_to_flow_mapping.keys():
                self.__tor_pair_to_flow_mapping[(src_tor_node, dst_tor_node)].append(flow_obj)
            else:
                self.__tor_pair_to_flow_mapping[(src_tor_node, dst_tor_node)] = [flow_obj]
            flow_obj.add_listener(self)
        self.__update_matrix()
        self._already_initialized = True

    def __init_matrix(self):
        """ Initializes the demand matrix with zeroes.
        """
        # (1) Get ToR Switches
        tor_switches = self._environment.topology.get_nodes_by_type("TorSwitch")
        assert len(tor_switches) > 0
        # (2) initialize 0s matrix
        self.__matrix = numpy.zeros((len(tor_switches), len(tor_switches)), dtype=numpy.dtype(Decimal))
        # (3) initalize node_id to matrix idx mapping
        for idx, rack in enumerate(tor_switches):
            assert rack not in self.__tor_node_id_to_idx_mapping.keys()
            self.__tor_node_id_to_idx_mapping[rack.node_id] = idx

    @property
    def matrix(self):
        return self.__matrix

    @property
    def has_flows(self):
        return numpy.any(self.__matrix)

    def __reset_matrix(self):
        """ Resets the matrix for recalculation. """
        self.__matrix = numpy.zeros(self.__matrix.shape, dtype=numpy.dtype(Decimal))

    def get_demand_by_node_ids(self, src_node_id, dst_node_id):
        """
        Returns the demand value by node ids or node instances. The demand is unidirectional
        so demand(1,2) is not necessarily demand(2,1).
        Args:
            src_node_id (str): node id of the src tor switch
            dst_node_id (str): node id of the dst tor switch

        Returns:
            demand (amount of traffic to be transmitted) between src_node_id and dst_node_id
        """

        return self.__matrix[
            self.__tor_node_id_to_idx_mapping[src_node_id],
            self.__tor_node_id_to_idx_mapping[dst_node_id]
        ]

    def get_demand_by_idx(self, src_idx, dst_idx):
        """
        Get the demand by matrix index
        Args:
            src_idx:
            dst_idx:

        Returns:
            the demand between tor with index src_idx and dst_idx
        """
        return self.__matrix[src_idx, dst_idx]

    def get_flows_by_tors(self, src_tor_switch_id, dst_tor_switch_id):
        """ Get the flows that contribute to the between src and dest. Unidirectional so
        the returned flows all go from src to dest.

        Returns:
            list of Flow objects
        """
        return self.__tor_pair_to_flow_mapping.get((src_tor_switch_id, dst_tor_switch_id), [])

    def get_bidirectional_flows_by_tors(self, src_tor_switch_id, dst_tor_switch_id):
        """ Same as get_flows_by_tors but is bidirectional so flows go from src to dest or the
        other way

        Returns:
            list of Flow objects
        """
        flows = list()
        if (src_tor_switch_id, dst_tor_switch_id) in self.__tor_pair_to_flow_mapping.keys():
            flows += (self.__tor_pair_to_flow_mapping[(src_tor_switch_id, dst_tor_switch_id)])
        if (dst_tor_switch_id, src_tor_switch_id) in self.__tor_pair_to_flow_mapping.keys():
            flows += (self.__tor_pair_to_flow_mapping[(dst_tor_switch_id, src_tor_switch_id)])
        return flows

    def __update_matrix(self):
        """ Iterates over all flows in the environment and extracts the demands between ToRs.
        Relies on a RotorNet topology structure to do so.
        Also maps flow ids to demands.
        """
        assert self.__matrix is not None
        self.__reset_matrix()
        for flows_of_tor_pair in self.__tor_pair_to_flow_mapping.values():
            for flow_obj in flows_of_tor_pair:
                src_tor_node, dst_tor_node = flow_handling.get_tor_node_ids_of_flow(flow_obj)
                # Add remaining volume to the demand entries
                self.__matrix[
                    self.__tor_node_id_to_idx_mapping[src_tor_node],
                    self.__tor_node_id_to_idx_mapping[dst_tor_node]
                ] += flow_obj.remaining_volume

    def notify_upd_time(self):
        """ Update the matrix if time advances."""
        # self.__update_matrix()
        pass

    def notify_add_flow(self, new_flow):
        """ Update the matrix if a flow is added."""
        src_tor_node, dst_tor_node = flow_handling.get_tor_node_ids_of_flow(new_flow)

        # Map the flow ids to a tor pair
        if (src_tor_node, dst_tor_node) in self.__tor_pair_to_flow_mapping.keys():
            self.__tor_pair_to_flow_mapping[(src_tor_node, dst_tor_node)].append(new_flow)
        else:
            self.__tor_pair_to_flow_mapping[(src_tor_node, dst_tor_node)] = [new_flow]

        self.__matrix[
            self.__tor_node_id_to_idx_mapping[src_tor_node],
            self.__tor_node_id_to_idx_mapping[dst_tor_node]
        ] += new_flow.remaining_volume
        # self.__update_matrix()

        new_flow.add_listener(self)

    def notify_rem_flow(self, removed_flow):
        """ Update the matrix if a flow is removed."""
        src_tor_node, dst_tor_node = flow_handling.get_tor_node_ids_of_flow(removed_flow)
        self.__tor_pair_to_flow_mapping[(src_tor_node, dst_tor_node)].remove(removed_flow)

        # No need to update matrix since remaining volume of flow should already be zero.
        # self.__update_matrix()
        if removed_flow.remaining_volume > Decimal('0'):
            self.__matrix[
                self.__tor_node_id_to_idx_mapping[src_tor_node],
                self.__tor_node_id_to_idx_mapping[dst_tor_node]
            ] -= removed_flow.remaining_volume
        removed_flow.remove_listener(self)

    def notify_update_remaining_volume(self, updated_flow, volume_change):
        """
        Reacts to updates of flow's remaining volume. Update the according entry in the matrix.
        Args:
            updated_flow: flow that was modified
            volume_change: the volume change

        Returns:

        """
        src_tor_node, dst_tor_node = flow_handling.get_tor_node_ids_of_flow(updated_flow)
        self.__matrix[
            self.__tor_node_id_to_idx_mapping[src_tor_node],
            self.__tor_node_id_to_idx_mapping[dst_tor_node]
        ] -= volume_change


@demand_matrix_factory_decorator
class SimpleTor2TorDemandMatrixWithFlowListFactory(object):
    def produce(self, environment, config):
        return SimpleTor2TorDemandMatrixWithFlowList(environment)


class SimpleTor2TorDemandMatrixWithFlowListConfiguration(configuration.AbstractConfiguration):
    def __init__(self):
        super(SimpleTor2TorDemandMatrixWithFlowListConfiguration, self).__init__(
            factory=SimpleTor2TorDemandMatrixWithFlowListFactory.__name__
        )

    @property
    def params(self):
        return self.__dict__
