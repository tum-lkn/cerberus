""" Flow factory for generation of different flow types """

from dcflowsim.environment.flow import Flow, NormalFlowFactory, SuperFlow, SuperFlowFactory

FLOW_FACTORIES = dict()
FLOW_FACTORIES[Flow.__name__] = NormalFlowFactory()
FLOW_FACTORIES[SuperFlow.__name__] = SuperFlowFactory()


def produce_flow(flow_type, global_id, arrival_time, source, destiation, volume):
    """ Selects a flow factory based on flow type attribute and produces a the corresponding
    Flow object"""
    return FLOW_FACTORIES[flow_type].produce(global_id=global_id,
                                             arrival_time=arrival_time,
                                             source=source,
                                             destination=destiation,
                                             volume=volume)
