ARRIVAL_TIME_GENERATOR_FACTORIES = dict()


def produce(atgen_configuration, rng=None):
    from . import arrival_time_generator as at_gen
    assert isinstance(atgen_configuration, at_gen.AbstractArrivalTimeGeneratorConfiguration)
    return ARRIVAL_TIME_GENERATOR_FACTORIES[atgen_configuration.factory].produce(
        configuration=atgen_configuration,
        rng=rng
    )
