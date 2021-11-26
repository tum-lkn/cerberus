import abc


class AbstractFlowScore(abc.ABC):

    def __init__(self, wrapped_flow):
        self.__flow = wrapped_flow

    @classmethod
    def dynamic(cls):
        raise NotImplementedError

    @property
    def flow(self):
        return self.__flow

    @abc.abstractmethod
    def score(self):
        """
        Class calculates scores_list for a flow.

        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Implement the comparison method. Could use the scores_list.

        Args:
            other: to be compared to

        Returns: True/False result based on comparison.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __lt__(self, other):
        """
        Implement the comparison method. Could use the scores_list.

        Args:
            other: to be compared to

        Returns: True/False results based on comparison

        """
        raise NotImplementedError

    def __repr__(self):
        return str(self.__flow)


class VolumeBasedFlowScore(AbstractFlowScore):
    """
    Class implements a volumne-based scoring. The larger the volume the higher the scores_list.
    """

    def __init__(self, wrapped_flow):
        super(VolumeBasedFlowScore, self).__init__(wrapped_flow)

    def score(self):
        return self.flow.volume

    @classmethod
    def dynamic(cls):
        return False

    def __eq__(self, other):
        return self.score() == other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    def __le__(self, other):
        return self.score() <= other.score()

    def __repr__(self):
        return "(VBFS:" + str(self.flow) + ")"


SCORE_FACTORY = dict()
SCORE_FACTORY[VolumeBasedFlowScore.__name__] = VolumeBasedFlowScore

