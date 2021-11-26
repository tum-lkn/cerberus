SWITCH_FACTORY = dict()


def produce_switch(config, node_id, num_ports, network_identifier):
    return SWITCH_FACTORY[config.__class__.__name__]().produce_switch(
        config, node_id, num_ports, network_identifier
    )
