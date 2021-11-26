STATISTIC_COLLECTOR_FACTORY = dict()


def statistic_collector_factory_decorator(factory=None):
    """
    Generic factory decorator. Takes the dict where the factory should be added, and if given,
    the factory name as well.

    Args:
        factory: the class from which we take the name

    Returns:
        the inner function returning the class

    """

    def inner(cls):
        STATISTIC_COLLECTOR_FACTORY[cls.__name__] = cls if factory is None else factory
        return cls

    return inner


NOTIFICATION_BEHAVIOR_FACTORY = dict()


def produce_notification_behavior(configuration):
    return NOTIFICATION_BEHAVIOR_FACTORY[configuration.factory].produce(configuration)


def produce(env, configuration):
    """

    Args:
        env:
        configuration:

    Returns:

    """
    # Build notification behavior
    not_behavior = produce_notification_behavior(configuration.notification_behavior_config)

    return STATISTIC_COLLECTOR_FACTORY[configuration.factory](env, not_behavior)
