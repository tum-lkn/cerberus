import abc
import logging

from dcflowsim import configuration
from dcflowsim.utils.factory_decorator import factory_decorator
from .statistic_collector_factory import NOTIFICATION_BEHAVIOR_FACTORY


class AbstractStatisticsNotificationBehavior(abc.ABC):
    @abc.abstractmethod
    def notify(self, listeners, notification_event, immediate=False):
        raise NotImplementedError


class PeriodicNotificationWithCollectorResetBehaviorConfiguration(configuration.AbstractConfiguration):
    def __init__(self, period):
        super(PeriodicNotificationWithCollectorResetBehaviorConfiguration, self). \
            __init__(PeriodicNotificationWithCollectorResetBehaviorFactory.__name__)
        self.__period = period

    @property
    def period(self):
        return self.__period

    @property
    def params(self):
        return self.__dict__


@factory_decorator(factory_dict=NOTIFICATION_BEHAVIOR_FACTORY)
class PeriodicNotificationWithCollectorResetBehaviorFactory(object):
    @classmethod
    def produce(cls, configuration):
        return PeriodicNotificationWithCollectorResetBehavior(configuration.period)


class PeriodicNotificationWithCollectorResetBehavior(AbstractStatisticsNotificationBehavior):
    """
    Forwards notification only every period-th event, i.e., this behavior "fires" every period interval.
    Resets the corresponding statistics collector after forwarding notifications
    """

    def __init__(self, period):
        """

        Args:
            period: number of received events after that this behavior forwards the event
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.__period = period
        self.__events_since_last_notification = 0

    def notify(self, listeners, notification_event, immediate=False):
        """
        Checks if period events  have been raised and forwards notification/event
        If immediate is set, this check is bypassed, the counter reset to 0 and the event is forwarded
        Args:
            listeners: list of environmentlistener objects
            notification_event: the event
            immediate: boolean flag

        Returns:

        """
        self.__events_since_last_notification += 1
        if (self.__events_since_last_notification >= self.__period) or immediate:
            self.__events_since_last_notification = 0
            for listener in listeners:
                self.logger.debug("Notifying {}. Event is {}".format(listener, notification_event))
                listener.notify(notification_event)

                # reset the index of the listener
                listener.reset_index()

            # reset the statistics collector
            notification_event.statistics_collector.notify_reset()
