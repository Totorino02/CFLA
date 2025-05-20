import torch
from torch.nn import Module
from framework.client.clientbase import Client


class ClientMADMTOP(Client):
    def __init__(self, client_id, train_loader, args, **kwargs):
        super().__init__(client_id, train_loader, args, **kwargs)
        self.local_model: Module = None
        self.train_loader = train_loader
        self.device = args["device"]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # args["criterion"]
        self.optimizer = args["optimizer"]
        self.local_epochs = args["local_epochs"]
        self.learning_rate = args["learning_rate"]
        self.history = []

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def get_params(self):
        return self.local_model.state_dict()

    def set_params(self, model_params: dict):
        self.local_model.load_state_dict(model_params)
