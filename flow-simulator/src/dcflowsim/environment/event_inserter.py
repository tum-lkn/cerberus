from . import environment_listener
from dcflowsim import constants
from dcflowsim.simulation import event


class AbstractEventInserter(environment_listener.EnvironmentListener):

    def __init__(self, environment, simulator):
        super(AbstractEventInserter, self).__init__(priority=constants.EVENT_INSERTION_PRIORITY)
        self._environment = environment
        self._simulator = simulator

    @property
    def environment(self):
        return self._environment


class EventInserter(AbstractEventInserter):

    def __init__(self, environment, simulator):
        super(EventInserter, self).__init__(environment, simulator)

    def initialize(self, flows):
        pass

    def add_event(self, event):
        self._simulator.add_event(event)

    def notify_upd_req_ocs(self, ocs, circuits):
        assert ocs.reconfig_delay > 0

        my_event = event.OcsReconfigurationEvent(
            self._simulator.environment.current_environment_time + ocs.reconfig_delay,
            ocs,
            circuits)

        self.add_event(my_event)
