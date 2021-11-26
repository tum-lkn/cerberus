import abc


class AbstractConfiguration(abc.ABC):
    @abc.abstractmethod
    def __init__(self, factory):
        self.__factory = factory

    @property
    def factory(self):
        return self.__factory

    @property
    @abc.abstractmethod
    def params(self):
        raise NotImplementedError
