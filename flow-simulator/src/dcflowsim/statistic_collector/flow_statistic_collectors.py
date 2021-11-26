import logging
from collections import namedtuple

import numpy as np

from .base_statistic_collector import AbstractStatisticCollector, StatisticsWriteSingleEvent
from .statistic_collector_factory import STATISTIC_COLLECTOR_FACTORY
from dcflowsim.utils.factory_decorator import factory_decorator

FlowCompletionTimeResult = namedtuple("FlowCompletionTimeResult", ["flow", "value", "first_time_with_rate"])


@factory_decorator(factory_dict=STATISTIC_COLLECTOR_FACTORY)
class SimpleFlowCompletionTimeCollector(AbstractStatisticCollector):
    """
    Collects Flow completion times along with the objects of the completed flows

    Implements a listener to the EnvironmentUpdate events.
    Further, has some listeners (data collectors) that it informs about new metric values
    """

    def __init__(self, environment=None, notification_behavior=None):
        super(SimpleFlowCompletionTimeCollector, self).__init__(notification_behavior)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__environment = environment
        self.__flow_completion_times = list()  # Stores FlowCompletionTimeResults

    @property
    def raw_data(self):
        return self.__flow_completion_times

    @property
    def flow_completion_times(self):
        return [fct_result.value for fct_result in self.__flow_completion_times]

    def initialize(self, flows):
        pass

    def notify_rem_flow(self, removed_flow):
        fct = self.__environment.current_environment_time - removed_flow.arrival_time
        self.__flow_completion_times.append(
            FlowCompletionTimeResult(
                removed_flow, float(fct), removed_flow.first_time_with_rate
            )
        )

        # Notify listeners
        self.notification_behavior.notify(
            self.listeners,
            StatisticsWriteSingleEvent(self)
        )

    def __repr__(self):
        return "{}(): current mean rates: {}".format(
            self.__class__.__name__,
            self.__flow_completion_times
        )

    def __str__(self):
        if len(self.__flow_completion_times) > 0:
            return "FCT ({} samples): min={} mean={} median={} max={} std={}".format(
                len(self.flow_completion_times),
                np.min(self.flow_completion_times),
                np.mean(self.flow_completion_times),
                np.median(self.flow_completion_times),
                np.max(self.flow_completion_times),
                np.std(self.flow_completion_times)
            )
        else:
            return self.__repr__()

    def print_statistics(self):
        self.logger.info(
            "FCT ({} samples): min={} mean={} median={} max={} std={}".format(
                len(self.flow_completion_times),
                np.min(self.flow_completion_times),
                np.mean(self.flow_completion_times),
                np.median(self.flow_completion_times),
                np.max(self.flow_completion_times),
                np.std(self.flow_completion_times)
            )
        )

    def notify_reset(self):
        self.__flow_completion_times = list()
