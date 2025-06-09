from collections import defaultdict

from framework.client.clientbase import Client
from framework.server.serverbase import Server
import numpy as np
import torch
import tqdm

class ServerFedAvg(Server):
    """
    This class implements the FedAvg algorithm for federated learning.
    It is a subclass of the Server class and uses the FedAvg algorithm to aggregate the updates from the clients.
    """

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.device = args["device"]
        self.criterion = args["criterion"]
        self.optimizer = args["optimizer"]
        self.local_epochs = args["local_epochs"]
        self.nb_epochs = args["nb_epochs"]
        self.learning_rate = args["learning_rate"]
        self.fraction = args["fraction"]
        self.history = []
        self.clients : list[Client]= []
        self.selected_clients : list[Client] = []
        self.train_loss_check = list()


    def train(self, verbose=False):
        for epoch in range(self.nb_epochs):
            self.global_model, loss = self.federated_learning(self.global_model, self.clients)
            if verbose:
                print(f"Epoch {epoch+1}/{self.nb_epochs} | Loss: {loss}")
            self.history.append(loss)
        return self.global_model, self.history


    def federated_learning(self, model, clients_subset):
        """
        This method performs federated learning on a subset of clients and returns the updated model
        and the mean loss of the local training.
        :param model: Model to be updated
        :param clients_subset: The subset of clients to perform federated learning on
        :return: An updated model, mean loss
        """
        # selects clients
        m = max(1, int(self.fraction * len(clients_subset)))
        selected_clients = np.random.choice(clients_subset, m, replace=False)

        total_samples = 0
        weighted_sum = None
        mean_loss = 0.0
        for client in selected_clients:
            data_size = len(client.train_loader.dataset)
            total_samples += data_size
            updated_params, loss = client.train(model)
            mean_loss += loss #* data_size
            self.train_loss_check.append(loss)
            if weighted_sum is None:
                weighted_sum = updated_params * data_size
            else:
                weighted_sum += updated_params * data_size

        # average the update vectors and loss
        mean_loss /= len(selected_clients)
        aggregated_update = (weighted_sum / total_samples).detach().clone()

        # update the global model
        offset = 0
        new_state_dict = {}
        for param_name, param in model.named_parameters():
            param_size = param.numel()
            delta = aggregated_update[offset: offset + param_size].view(param.size())
            new_state_dict[param_name] = delta.clone()
            offset += param_size
        model.load_state_dict(new_state_dict)
        return model, mean_loss

    def aggregate(self, client_updates):
        """
        This method aggregates the updates from the clients using the FedAvg algorithm.
        :param client_updates: A list of client updates
        :return: The aggregated model
        """
        pass

    def evaluate(self, **kwargs):
        pass

    def set_clients(self, clients: list[Client]):
        self.clients = clients

    def get_params(self):
        pass

