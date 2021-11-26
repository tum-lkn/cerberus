SIMULATION_BEHAVIOR_FACTORY = dict()


def simulation_behavior_factory_decorator(cls):
    """
    Decorator function for simulation behavior factories.

    Returns:
        the decorated class

    """

    SIMULATION_BEHAVIOR_FACTORY[cls.__name__] = cls
    return cls


def produce(behavior_configuration, simulation_configuration, traffic_generator):
    return SIMULATION_BEHAVIOR_FACTORY[behavior_configuration.factory]().generate_behavior(
        behavior_configuration,
        simulation_configuration,
        traffic_generator
    )
