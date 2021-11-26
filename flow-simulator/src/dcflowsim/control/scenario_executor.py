import abc
import logging

from dcflowsim.control import task


class LocalTaskExecutor(object):

    def __init__(self):
        super(LocalTaskExecutor, self).__init__()

        self.logger = logging.getLogger(
            self.__class__.__name__
        )

        self.sim_cfgs = list()

    def add_simulation_config(self, sim_cfg):
        """
        Add sim cfg to local task execute. Just store the sim_cfg.

        Args:
            sim_cfg: to be added

        Returns: None

        """
        self.sim_cfgs.append(sim_cfg)

    @abc.abstractmethod
    def run(self):
        """
        Run tasks locally.

        Returns: None

        """
        self.logger.debug("Running {} simulations.".format(
            len(self.sim_cfgs)
        ))
        for sim_cfg in self.sim_cfgs:
            task.run_simulation(sim_cfg)
        self.logger.debug("Finished!")
        return True
