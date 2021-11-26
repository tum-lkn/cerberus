STATISTIC_LISTENER_FACTORY = dict()


def produce(configuration, simulation_builder):
    return STATISTIC_LISTENER_FACTORY[configuration.factory]().produce(
        configuration,
        simulation_builder
    )
