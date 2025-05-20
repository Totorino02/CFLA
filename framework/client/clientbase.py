from abc import ABC, abstractmethod

class Client(ABC):
    """
    This is the base class for a client.
    For each type of algorithm, we would inherit from this class and implement methods
    """
    def __init__(self, client_id, train_loader, args, **kwargs):
        self.client_id = client_id
        self.train_loader = train_loader
        self.optimizer = args["optimizer"]
        self.criterion = args["criterion"]
        self.nb_epochs = 0

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass
