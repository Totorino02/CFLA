import torch
from framework.client.client_flhc import ClientFLHC
from framework.server.serverbase import Server
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class ServerFLHC(Server):

    def __init__(self, global_model, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.fraction = args["fraction"]
        self.device = args["device"]
        self.initial_rounds = args["initial_rounds"]
        self.cluster_rounds = args["cluster_rounds"]
        self.clusters = None
        self.clients : list[ClientFLHC]= []
        self.selected_clients : list[ClientFLHC] = []
        self.history = []
        self.specialized_models = dict()


    def pre_learning(self):
        for epoch in range(self.initial_rounds):
            self.global_model = self.federated_learning(self.global_model, self.clients)

    def federated_learning(self, model, clients_subset):
        # selects clients
        m = max(1, int(self.fraction * len(clients_subset)))
        selected_clients = np.random.choice(clients_subset, m, replace=False)

        total_samples = 0
        weighted_sum = None
        for client in selected_clients:
            data_size = len(client.train_loader.dataset)
            total_samples += data_size
            update_vector = client.train(model)

            if weighted_sum is None:
                weighted_sum = update_vector * data_size
            else:
                weighted_sum += update_vector * data_size

        # average the update vectors
        aggregated_update =  torch.tensor(weighted_sum / total_samples)

        # update the global model
        offset = 0
        new_state_dict = {}

        for param_name, param in model.named_parameters():
            param_size = param.numel()
            delta = aggregated_update[offset: offset + param_size].view(param.size())
            new_state_dict[param_name] = (param.data.cpu() + delta).clone()
            offset += param_size

        model.load_state_dict(new_state_dict)
        return model

    def hierarchical_clustering(self, updates, threshold=1.0, metric="euclidean"):
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric=metric,
            linkage='complete'
        )

        labels = clustering.fit_predict(updates)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(self.clients[idx])
        return clusters

    def train(self):
        # Pre-learning
        self.pre_learning()

        # hierarchical clustering
        updates = []
        for client in self.clients:
            client_update = client.train(self.global_model)
            updates.append(client_update)
        updates = np.stack(updates)
        self.clusters = self.hierarchical_clustering(updates)

        # personalized models
        specialized_models = {}
        for label, cluster_clients in self.clusters.items():
            cluster_model = type(self.global_model)().to(self.device)
            cluster_model.load_state_dict(self.global_model.state_dict())
            for _ in range(self.cluster_rounds):
                cluster_model = self.federated_learning(cluster_model, cluster_clients)
            specialized_models[label] = cluster_model
        self.specialized_models = specialized_models
        return specialized_models

    def evaluate(self):
        pass

    def aggregate(self):
        pass

    def set_clients(self, clients: list[ClientFLHC]):
        self.clients = clients

    def get_params(self):
        return self.global_model.state_dict()

