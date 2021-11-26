ALGORITHM_FACTORIES = dict()


def algorithm_factory_decorator(cls):
    """
    Class decorator for all algorithm factories.

    Args:
        cls: the factory class that should be decorated.

    Returns:
        simply the cls
    """
    ALGORITHM_FACTORIES[cls.__name__] = cls
    return cls


def produce(algorithm_configuration, env):
    alg = ALGORITHM_FACTORIES[algorithm_configuration.factory]().generate_algorithm(
        algorithm_configuration,
        env
    )
    return alg
