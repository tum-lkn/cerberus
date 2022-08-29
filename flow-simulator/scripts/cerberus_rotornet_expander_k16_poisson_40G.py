import demand_configs
from dcflowsim import logger
from dcflowsim.algorithm import rotornet_algorithm, score_based_algorithm, flow_score, cerberus_algorithm, \
    cerberus_flow_distribution_strategy
from dcflowsim.control import scenario_executor
from dcflowsim.data_writer import database_statistics_writer
from dcflowsim.environment import demand_matrix, flow
from dcflowsim.flow_generation import random_flow_generator
from dcflowsim.network.cerberus import *
from dcflowsim.network.expandernet import *
from dcflowsim.network.network_elements import *
from dcflowsim.network.rotornet import *
from dcflowsim.simulation import simulation, simulation_builder, simulation_behavior
from dcflowsim.statistic_collector import notification_behavior, base_statistic_collector, flow_statistic_collectors

if __name__ == "__main__":
    logger.init_logging("logging.yaml")

    TRAFFIC_DURATION = 5  # in s

    NR_HOST_PER_RACK = 1
    NR_RACKS = 64
    OPTICAL_LINK_CAPACITY = 40000
    TIME_SCALING = 1e6

    K = 16
    tor_switch_capacity = K * OPTICAL_LINK_CAPACITY
    ROTOR_DAY = 90
    ROTOR_NIGHT = 10

    TOPO_CERBERUS = CerberusTopologyConfiguration(
        nr_hosts_per_rack=NR_HOST_PER_RACK,
        nr_racks=NR_RACKS,
        link_capacity=OPTICAL_LINK_CAPACITY,
        nr_static=0, nr_rotor=4, nr_da=12,
        rotor_day_duration=ROTOR_DAY,
        rotor_night_duration=ROTOR_NIGHT,
        ocs_reconfig_delay=15_000,
        tor_switch_config=ElectricalSwitchWithOpticalPortsConfiguration(K, OPTICAL_LINK_CAPACITY),
        seed=0
    )

    TOPO_ROTORNET = RotorNetTopologyConfiguration(
        nr_hosts_per_rack=NR_HOST_PER_RACK,
        nr_racks=NR_RACKS,
        tor_switch_config=ElectricalSwitchWithOpticalPortsConfiguration(K, tor_switch_capacity),
        nr_rotorswitches=K,
        rotor_switch_configs=get_rotorswitch_configs(
            nr_racks=NR_RACKS,
            nr_rotorswitches=K,
            day_duration=ROTOR_DAY, night_duration=ROTOR_NIGHT,
            link_capacity=OPTICAL_LINK_CAPACITY,
        ),
        optical_link_capacity=OPTICAL_LINK_CAPACITY
    )

    TOPO_EXPANDER = ExpanderNetTopologyConfiguration(
        nr_hosts_per_rack=NR_HOST_PER_RACK,
        nr_racks=NR_RACKS,
        nr_links=K,
        tor_switch_config=ElectricalSwitchConfiguration(OPTICAL_LINK_CAPACITY),
        seed=0,
        host_link_capacity=tor_switch_capacity
    )

    ALGO_ROTORNET = rotornet_algorithm.RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationConfiguration(
        rate_limit_factor=1)

    ALGO_EXPANDERNET = score_based_algorithm.TorByTorGreedyScoreBasedAlgorithmConfiguration(
        score_factory=flow_score.VolumeBasedFlowScore.__name__
    )

    static_algo = score_based_algorithm.TorByTorGreedyScoreBasedAlgorithmConfiguration(
        score_factory=flow_score.VolumeBasedFlowScore.__name__,
        topology_view_configuration=ocs_topology.TopologyWithOcsViewConfiguration(
            relevant_link_identifiers=[LinkIdentifiers.static, LinkIdentifiers.default],
            relevant_ocs_identifiers=[]
        )
    )
    rnet_algo = rotornet_algorithm.RotorNetTorToTorTwoHopAlgorithmWithFullRateAllocationConfiguration(
        tor2tor_demand_matrix_configuration=demand_matrix.SimpleTor2TorDemandMatrixWithFlowListConfiguration(),
        topology_view_configuration=ocs_topology.TopologyWithOcsViewConfiguration(
            relevant_link_identifiers=[LinkIdentifiers.rotor, LinkIdentifiers.default],
            relevant_ocs_identifiers=[ocs_topology.OCSSwitchTypes.rotor]
        ),
        rate_limit_factor=1
    )
    da_algo = score_based_algorithm.GreedyOpticalScoreBasedAlgorithmConfiguration(
        score_factory=flow_score.VolumeBasedFlowScore.__name__,
        topology_view_configuration=ocs_topology.TopologyWithOcsViewConfiguration(
            relevant_link_identifiers=[LinkIdentifiers.dynamic,
                                       LinkIdentifiers.default],
            relevant_ocs_identifiers=[ocs_topology.OCSSwitchTypes.ocs]
        ),
        raise_no_allocation=True,
        allow_multi_allocation=False
    )

    ALGO_CERBERUS = cerberus_algorithm.CerberusAlgorithmConfiguration(
        static_algorithm=static_algo, rotornet_algorithm=rnet_algo, da_algorithm=da_algo,
        flow_distribution_strategy_configuration=
        cerberus_flow_distribution_strategy.FixedThresholdsDynamicFlowDistributionStrategyConfiguration(
            medium_threshold=0, large_threshold=1.7e8
        )
    )

    # Base number of flows for bidi load with 64 racks and with 40G ports
    BASE_NUMBER_FLOWS_BIDI = 5088 * 4
    loads_to_sweep = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    flow_generation_configurations = list()
    for load in loads_to_sweep:
        num_flows = int(load * BASE_NUMBER_FLOWS_BIDI * K)
        flow_generation_configurations.append(
            random_flow_generator.RandomWithConnectionProbabilityFlowGeneratorConfiguration(
                seed=0,
                max_num_flows=num_flows * TRAFFIC_DURATION,
                volume_generation_configuration=demand_configs.DATAMINING_VOLUMES,
                arrival_time_generation_configuration=demand_configs.get_poisson_at(num_flows),
                connection_generation_configuration=demand_configs.UNIFORM_HOST_CONNECTIONS,
                flow_type=flow.SuperFlow.__name__
            ))

    expandernet_flow_generation_configurations = list()

    for load in loads_to_sweep:
        num_flows = int(load * BASE_NUMBER_FLOWS_BIDI * K)
        expandernet_flow_generation_configurations.append(
            random_flow_generator.RandomWithConnectionProbabilityFlowGeneratorConfiguration(
                seed=0,
                max_num_flows=num_flows * TRAFFIC_DURATION,
                volume_generation_configuration=demand_configs.DATAMINING_VOLUMES,
                arrival_time_generation_configuration=demand_configs.get_poisson_at(num_flows),
                connection_generation_configuration=demand_configs.UNIFORM_HOST_CONNECTIONS,
            )
        )

    statistics_configuration = [
        base_statistic_collector.StatisticCollectorConfiguration(
            factory=flow_statistic_collectors.SimpleFlowCompletionTimeCollector.__name__,
            notification_behavior_config=
            notification_behavior.PeriodicNotificationWithCollectorResetBehaviorConfiguration(1000)
        )
    ]

    db_config = {
        'host': "cerberus-db",
        'port': 3306,
        'user': "root",
        'password': "passwort_simulator",
        'database': "cerberus"
    }
    data_model_configuration = [
        database_statistics_writer.FlowCompletionTimeDatabaseWriterConfiguration(
            db_config=db_config
        )
    ]

    rotornet_behavior = simulation_behavior.SimulationBehaviorWithDatabaseCheckConfiguration(
        decorated_behavior=simulation_behavior.RotornetSimulationBehaviorSparseArrivalsConfiguration(
            day_duration=ROTOR_DAY, night_duration=ROTOR_NIGHT
        ),
        db_config=db_config
    )

    expandernet_behavior = simulation_behavior.SimulationBehaviorWithDatabaseCheckConfiguration(
        decorated_behavior=simulation_behavior.SimulationBehaviorSparseArrivalsConfiguration(),
        db_config=db_config
    )

    simulations = list()

    sim_time = 18.5e6
    simulation_builder = simulation_builder.SimpleSimulationBuilderWithEventInsertion
    scenario_executor = scenario_executor.LocalTaskExecutor()

    for flowgen in flow_generation_configurations:
        simulations.append(
            simulation.BasicSimulatorConfiguration(
                factory=simulation_builder,
                simulation_time=sim_time,
                topology_configuration=TOPO_ROTORNET,
                behavior_configuration=rotornet_behavior,
                algorithm_configuration=ALGO_ROTORNET,
                flow_generation_configuration=flowgen,
                statistics_configuration=statistics_configuration,
                data_model_configuration=data_model_configuration
            )
        )

        simulations.append(
            simulation.BasicSimulatorConfiguration(
                factory=simulation_builder,
                simulation_time=sim_time,
                topology_configuration=TOPO_CERBERUS,
                behavior_configuration=rotornet_behavior,
                algorithm_configuration=ALGO_CERBERUS,
                flow_generation_configuration=flowgen,
                statistics_configuration=statistics_configuration,
                data_model_configuration=data_model_configuration
            )
        )

    for flowgen in expandernet_flow_generation_configurations:
        simulations.append(
            simulation.BasicSimulatorConfiguration(
                factory=simulation_builder,
                simulation_time=sim_time,
                topology_configuration=TOPO_EXPANDER,
                behavior_configuration=expandernet_behavior,
                algorithm_configuration=ALGO_EXPANDERNET,
                flow_generation_configuration=flowgen,
                statistics_configuration=statistics_configuration,
                data_model_configuration=data_model_configuration
            )
        )

    print(f'Created {len(simulations)} simulation configurations.')

    for config in simulations:
        simulation_builder(config).build_simulation()
        scenario_executor.add_simulation_config(config)
    scenario_executor.run()
    print("Done")
