from dcflowsim import constants

PEEWEE_INTERFACE_FACTORY = dict()


def peewee_interface_factory_decorator(interface=None):
    """
    Class decorator for all peewee interface factories.

    Args:
        interface: the factory class that should be decorated.

    Returns:
        simply the class
    """
    assert interface is not None

    def inner(cls):
        PEEWEE_INTERFACE_FACTORY[cls.__name__] = interface
        return cls

    return inner


INTERFACE_FACTORY = dict()
INTERFACE_FACTORY[constants.INTERFACE_TYPE_DATABASE] = PEEWEE_INTERFACE_FACTORY


def produce_interface(entity, connection_manager):
    """
    Returns correct peewee interface for given entity
    Args:
        entity:
        connection_manager:

    Returns:
        instance of AbstractPeeweeInterface
    """
    entity_type = type(entity)
    return INTERFACE_FACTORY[connection_manager.interface_type][entity_type.__name__](connection_manager.connection)
