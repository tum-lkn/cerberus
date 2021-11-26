TOPOLOGY_FACTORIES = dict()


def produce(topology_configuration, decorated_topology=None):
    return TOPOLOGY_FACTORIES[topology_configuration.factory]().generate_topology(
        topology_configuration,
        decorated_topology
    )


class TopologyBuilder(object):
    @classmethod
    def build_topologies(cls, topology_configuration):

        if topology_configuration.decorated_topology_configuration is not None:
            decorated_topology = TopologyBuilder.build_topologies(
                topology_configuration.decorated_topology_configuration
            )
        else:
            decorated_topology = None
        return produce(topology_configuration, decorated_topology)


TOPOLOGY_VIEW_FACTORIES = dict()


def produce_view(topology_view_configuration, original_topology):
    if topology_view_configuration is not None:
        return TOPOLOGY_VIEW_FACTORIES[topology_view_configuration.factory]().produce(
            topology_view_configuration, original_topology
        )
    else:
        return original_topology
