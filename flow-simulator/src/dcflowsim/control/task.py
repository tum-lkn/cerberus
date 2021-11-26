import logging


def run_simulation(simulation_config):
    task_logger = logging.getLogger(__name__)
    task_logger.info("Received simulation config")
    simulator = simulation_config.factory(
        simulation_config
    ).build_simulation()
    task_logger.info("Built simulator")
    simulator.run()
    task_logger.info("Finished simulation")
    return True
