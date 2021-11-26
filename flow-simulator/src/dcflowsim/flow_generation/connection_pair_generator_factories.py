CONNECTION_PAIR_GENERATOR_FACTORY = dict()


def produce_connection_generator(congen_configuration, environment, rng=None):
    return CONNECTION_PAIR_GENERATOR_FACTORY[congen_configuration.factory]().produce(
        config=congen_configuration,
        environment=environment,
        rng=rng
    )
