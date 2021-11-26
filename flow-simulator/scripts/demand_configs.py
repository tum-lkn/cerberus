from dcflowsim.flow_generation import arrival_time_generator, connection_pair_generator, flow_volume_generator

datamining_values = \
        [100, 180, 250, 560, 900, 1100, 1870, 3160, 10000, 100001, 400000, 1850000, 10000000, 30000000, 100000000,
         250000000, 1000000000]

DATAMINING_VOLUMES = flow_volume_generator.CdfBasedFlowVolumeGeneratorConfiguration(
    values=[8 * item for item in datamining_values],
    cdf_values=[0, 0.085, 0.14, 0.33, 0.47, 0.55, 0.65, 0.7, 0.8, 0.874, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1],
    seed=None
)

UNIFORM_HOST_CONNECTIONS = connection_pair_generator.HostLevelTorConnectionPairUniformGeneratorConfiguration(
        tor_switch_identifier="TorSwitch")


def get_poisson_at(num_flows_per_sec):
    return arrival_time_generator.PoissonArrivalTimeGeneratorConfiguration(
        mean_inter_arrival_time=1e6/num_flows_per_sec,   # We currently use usecs as simulation time unit
        seed=None       # Seed is anyways overwritten.
    )
