import torch
from torch.nn import Module
from framework.client.clientbase import Client
import numpy as np


class ClientMADMTOP(Client):
    def __init__(self, client_id, dataset, args, **kwargs):
        super().__init__(client_id, dataset, args, **kwargs)
        self.local_model: Module = None
        self.dataset = dataset
        self.device = args["device"]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # args["criterion"]
        self.optimizer = args["optimizer"]
        self.local_epochs = args["local_epochs"]
        self.learning_rate = args["learning_rate"]
        self.history = []

    def train(self, global_model=None, verbose=False, **kwargs):
        """
                This method trains the local model on the local dataset
                and returns the updated vector (the difference between the global model params and the local trained model params)
                 and the last training loss.
                :param global_model: The global model gets from the server
                :param verbose: If True, prints the training progress
                :return: Update_vector, loss
                """
        # copy of the global model to a local model
        self.local_model = type(global_model)().to(self.device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.train()
        self.optimizer = torch.optim.SGD(params=self.local_model.parameters(), lr=self.learning_rate)

        # training on local data
        loss = torch.tensor(0.0)
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                self.history.append(loss.item())
                loss.backward()
                self.optimizer.step()

            if verbose:
                if epoch % 10 == 0:
                    print(f"Client {self.client_id} | Epoch {epoch + 1}/{self.local_epochs} | Loss: {loss.item()}")

        # update vector of local model parameters
        delta_params = []
        updated_params = []
        empirical_risk_grad = []
        for new_param, old_param in zip(self.local_model.parameters(), global_model.parameters()):
            delta_params.append((new_param.data - old_param.data).flatten())
            updated_params.append(new_param.data.flatten())
            if new_param.grad is not None:
                empirical_risk_grad.append(new_param.grad.detach().clone().flatten())

        delta_params = torch.cat(delta_params)
        updated_params = torch.cat(updated_params)
        empirical_risk_grad = torch.cat(empirical_risk_grad)#.numpy()
        return updated_params, delta_params, empirical_risk_grad, loss.item()

    def test(self):
        pass

    def predict(self):
        pass

    def get_params(self):
        return self.local_model.state_dict()

    def set_params(self, model_params: dict):
        self.local_model.load_state_dict(model_params)
