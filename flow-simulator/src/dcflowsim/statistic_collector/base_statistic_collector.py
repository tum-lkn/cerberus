import logging
from abc import abstractmethod

from dcflowsim import configuration, constants
from dcflowsim.environment import environment_listener
from .notification_behavior import PeriodicNotificationWithCollectorResetBehavior


class StatisticCollectorConfiguration(configuration.AbstractConfiguration):

    def __init__(self, factory, notification_behavior_config):
        super(StatisticCollectorConfiguration, self).__init__(
            factory=factory
        )
        self.__notification_behavior_config = notification_behavior_config

    @property
    def notification_behavior_config(self):
        return self.__notification_behavior_config

    @property
    def params(self):
        return self.__dict__


class AbstractStatisticsUpdateEvent(object):
    """
    Abstract class for statistics listener events.
    """

    @abstractmethod
    def visit(self, listener):
        raise NotImplementedError


class StatisticsWriteSingleEvent(AbstractStatisticsUpdateEvent):
    """
    Event to notify StatisticListeners that there is new data available to consume
    """

    def __init__(self, statistics_collector):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__statistics_collector = statistics_collector

    def visit(self, listener):
        listener.notify_write(self.__statistics_collector)

    @property
    def statistics_collector(self):
        return self.__statistics_collector

    def __str__(self):
        return "StatisticsWriteSingle"


class AbstractStatisticCollector(environment_listener.EnvironmentListener):
    def __init__(self, notification_behavior=None):
        super(AbstractStatisticCollector, self).__init__(priority=constants.STATISTICS_COLLECTOR_PRIORITY)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.notification_behavior = notification_behavior
        if self.notification_behavior is None:
            self.notification_behavior = PeriodicNotificationWithCollectorResetBehavior(100)
        self.__statistics_listeners = list()

    def add_statistics_listener(self, listener):
        self.__statistics_listeners.append(listener)
        self.logger.info("Added statistics listener {}".format(listener))

    @property
    def listeners(self):
        return self.__statistics_listeners

    @abstractmethod
    def print_statistics(self):
        raise NotImplementedError
