FLOW_GENERATOR_FACTORIES = dict()


def produce_flow_generator(flowgen_configuration, environment):
    return FLOW_GENERATOR_FACTORIES[flowgen_configuration.factory]().produce(
        configuration=flowgen_configuration,
        environment=environment
    )
