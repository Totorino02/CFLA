from abc import ABC, abstractmethod

from framework.client.clientbase import Client


class Server(ABC):
    def __init__(self):
        self.clients : list[Client] = []
        self.clusters : dict[int, list] = dict()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def aggregate(self, **kwargs):
        pass

    @abstractmethod
    def get_params(self):
        pass



