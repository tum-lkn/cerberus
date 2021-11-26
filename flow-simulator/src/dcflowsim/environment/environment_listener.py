import abc
import logging


class EnvironmentListener(abc.ABC):
    """
    Abstract Environment listener implements subscriber to environment
    """

    def __init__(self, priority):
        """
        Priority of the listener. Determines the order in which environment listeners are notified. Low value means
        early notification
        Args:
            priority:
        """
        self.__priority = priority
        self.logger = logging.getLogger(EnvironmentListener.__class__.__name__)

    @property
    def priority(self):
        return self.__priority

    def initialize(self, flows):
        """
        Initializes the environment listener. For instance, can be used to perform an initial sync with the state of
        the environment
        Args:
            flows:

        Returns:

        """
        raise NotImplementedError

    def get_sub_listeners(self):
        """
        Returns a list of EnvironmentListeners that are actively used by this EnvironmentListener
        Returns:
            list of EnvironmentListeners
        """
        raise NotImplementedError

    def notify(self, notification_event):
        """

            Cases:
                ArrivalEvent
                ServiceEvent

            Args:
                notification_event:

            Returns:

        """
        notification_event.visit(self)

    def notify_add_flow(self, new_flow):
        pass

    def notify_rem_flow(self, removed_flow):
        pass

    def notify_partial_rem_flow(self, removed_flow):
        pass

    def notify_upd_time(self):
        pass

    def notify_upd_ocs(self, ocs):
        pass

    def notify_upd_req_ocs(self, ocs, circuits):
        pass

    def notify_upd_flow_route(self, changed_flow):
        pass

    def notify_scheduled_call(self, env_listener):
        """
        Scheduled call.
        """
        pass

    def notify_reset(self):
        self.logger.info("Resetting but this object is stateless")

    def __lt__(self, other):
        return self.priority < other.priority
