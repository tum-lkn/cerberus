import abc

from dcflowsim import configuration


class AbstractStatisticListener(abc.ABC):
    """
    Abstract StatisticListener. Listener subscribed to a statistics collector (database writer or file writer in
    current usage)
    """

    def __init__(self):
        pass

    def notify(self, notification_event):
        """
            Cases:
                StatisticsWriteSingleEvent: Event to notify StatisticListeners that there is new data
                available to consume

            Args:
                notification_event: AbstractStatisticsUpdateEvent
        """
        notification_event.visit(self)

    @abc.abstractmethod
    def notify_write(self, statistics_collector):
        pass


class StatisticListener(AbstractStatisticListener):
    """
    Abstract StatisticListener. Listener subscribed to a statistics collector (database writer or file writer in
    current usage). Needed for tests as abstract classes can't be instantiated with abstract methods
    """

    def __init__(self):
        super(StatisticListener, self).__init__()

    def notify(self, notification_event):
        """
            Cases:
                StatisticsWriteSingleEvent: Event to notify StatisticListeners that there is new data
                available to consume

            Args:
                notification_event: AbstractStatisticsUpdateEvent
        """
        notification_event.visit(self)

    def notify_write(self, statistics_collector):
        pass


class StatisticListenerConfiguration(configuration.AbstractConfiguration):
    def __init__(self, factory, metric):
        """

        Args:
            factory:
            metric: name of StatisticCollector class that this StatisticListener should listen on
        """
        super(StatisticListenerConfiguration, self).__init__(
            factory=factory
        )
        self.__metric = metric

    @property
    def metric(self):
        return self.__metric

    @property
    def params(self):
        return self.__dict__
