VOLUME_GENERATOR_FACTORIES = dict()


def produce(volgen_configuration, rng=None):
    from . import flow_volume_generator as fvgen
    assert isinstance(volgen_configuration, fvgen.AbstractFlowVolumeGeneratorConfiguration)
    return VOLUME_GENERATOR_FACTORIES[volgen_configuration.factory]().produce(
        configuration=volgen_configuration,
        rng=rng
    )
