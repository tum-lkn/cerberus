from abc import ABC, abstractmethod

from dcflowsim import configuration


class AbstractAlgorithmConfiguration(configuration.AbstractConfiguration):

    def __init__(self, factory):
        super(AbstractAlgorithmConfiguration, self).__init__(factory=factory)

    @property
    def params(self):
        raise NotImplementedError


class AbstractAlgorithmFactory(ABC):
    """
    A class implementing a topology factory.
    """

    @classmethod
    @abstractmethod
    def generate_algorithm(cls, algorithm_configuration, env):
        raise NotImplementedError
