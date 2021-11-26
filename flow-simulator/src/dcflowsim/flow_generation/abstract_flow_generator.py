import abc


class AbstractFlowGeneratorConfiguration(abc.ABC):
    def __init__(self, factory):
        self.factory = factory
