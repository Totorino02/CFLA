from collections import defaultdict

import torch
from framework.client.client_flhc import ClientFLHC
from framework.server.serverbase import Server
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import time
from framework.common.utils import convert_seconds_to_hhmmss
from tqdm import tqdm

class ServerFLHC(Server):

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.fraction = args["fraction"]
        self.device = args["device"]
        self.initial_rounds = args["initial_rounds"]
        self.cluster_rounds = args["cluster_rounds"]
        self.distance_threshold = args["distance_threshold"]
        self.clustering_metric = args["clustering_metric"]
        self.clusters = None
        self.clients : list[ClientFLHC]= []
        self.selected_clients : list[ClientFLHC] = []
        self.history = []
        self.specialized_models = dict()


    def pre_learning(self):
        for epoch in range(self.initial_rounds):
            self.global_model, loss = self.federated_learning(self.global_model, self.clients)

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
            updated_params, update_vector, loss = client.train(model)
            mean_loss += loss
            if weighted_sum is None:
                weighted_sum = updated_params * data_size
            else:
                weighted_sum += updated_params * data_size

        # average the update vectors and loss
        mean_loss /= m
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

    def hierarchical_clustering(self, updates):
        """
        This method performs hierarchical clustering on the update vectors of the clients and returns the clusters.
        :param updates: Update vectors
        :return: Clusters
        """
        clustering = AgglomerativeClustering(
            n_clusters=2,
            #distance_threshold=self.distance_threshold,
            #metric=self.clustering_metric,
            linkage='complete'
        )

        labels = clustering.fit_predict(updates)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(self.clients[idx])
        return clusters

    def train(self):
        """
        This method trains the global model on the clients' datasets and returns the specialized models.
        The training is done in the following steps:
            1. Pre-learning: The global model is trained on the clients' datasets for a fixed number of rounds.
            2. Clustering: The update vectors of the clients are clustered using hierarchical clustering.
            3. Personalization: The specialized models are trained on each cluster's dataset for a fixed number of rounds.
        :return: Specialized models and cluster's mean loss history
        """
        training_start_time = time.time()

        # Pre-learning
        print("Pre-learning starts...")
        pretrain_start_time = time.time()
        self.pre_learning()
        pretrain_end_time = time.time()
        print("Pre-learning ends.\n")

        # hierarchical clustering
        updates = []
        for client in self.clients:
            _, client_update, loss = client.train(self.global_model)
            updates.append(client_update)
        updates = np.stack(updates)
        print("Clustering starts...")
        cluster_start_time = time.time()
        self.clusters = self.hierarchical_clustering(updates)
        cluster_end_time = time.time()
        print(f"Clustering ends... Nb of clusters : {len(self.clusters)}\n")

        # personalized models

        specialized_models = dict()
        training_loss_history = defaultdict(list)
        test_loss_history = defaultdict(list)
        accuracy_history = defaultdict(list)
        global_test_accuracy_history = defaultdict(list)
        global_test_loss_history = defaultdict(list)
        print("Specialized models training...")
        personalization_start_time = time.time()
        for label, cluster_clients in self.clusters.items():
            cluster_model = type(self.global_model)().to(self.device)
            cluster_model.load_state_dict(self.global_model.state_dict())
            print(f"Cluster {int(label)+1} training...")
            for _ in tqdm(range(self.cluster_rounds), unit="round", colour="green"):
                cluster_model, mean_loss = self.federated_learning(cluster_model, cluster_clients)
                training_loss_history[int(label)].append(mean_loss)
                # pick a random client int the cluster and perform the eval
                client_id = np.random.randint(0, len(cluster_clients))
                acc_top1, acc_topk, test_loss = self.evaluate(cluster_model, cluster_clients[client_id].test_loader, k=1, return_loss=True)
                g_acc_top1, g_acc_topk, g_test_loss = self.evaluate(cluster_model, self.test_dataloader, k=1, return_loss=True)
                test_loss_history[int(label)].append(test_loss)
                accuracy_history[int(label)].append(acc_top1)
                global_test_accuracy_history[int(label)].append(g_acc_top1)
                global_test_loss_history[int(label)].append(g_test_loss)
            specialized_models[label] = cluster_model
        self.specialized_models = specialized_models
        training_end_time = time.time()
        personalization_end_time = time.time()
        print("Specialized models end training.\n")
        print(f" ---- STATS ----- : ")
        print(f"Pre-training time : {convert_seconds_to_hhmmss(pretrain_end_time - pretrain_start_time)}")
        print(f"Clustering time : {convert_seconds_to_hhmmss(cluster_end_time - cluster_start_time)}")
        print(f"Personalization time : {convert_seconds_to_hhmmss(personalization_end_time - personalization_start_time)}")
        print(f"Training time : {convert_seconds_to_hhmmss(training_end_time - training_start_time)}")
        print(f"---- END STATS ----- ")
        return specialized_models, training_loss_history, test_loss_history, accuracy_history, global_test_accuracy_history, global_test_loss_history

    def evaluate(self, model, test_dataloader,  k=5, return_loss=False):
        model.eval()
        correct_top1 = 0
        correct_topk = 0
        total = 0
        total_loss = 0.0
        last_lost = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Top-1
                _, pred_top1 = outputs.max(dim=1)
                correct_top1 += (pred_top1 == targets).sum().item()

                # Top-k
                _, pred_topk = outputs.topk(k, dim=1)
                correct_topk += pred_topk.eq(targets.view(-1, 1)).sum().item()

                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                last_lost = loss.item()

        acc_top1 = correct_top1 / total
        acc_topk = correct_topk / total
        avg_loss = total_loss / total

        if return_loss:
            return acc_top1, acc_topk, last_lost
        else:
            return acc_top1, acc_topk

    def aggregate(self):
        pass

    def set_clients(self, clients: list[ClientFLHC]):
        self.clients = clients

    def get_params(self):
        return self.global_model.state_dict()

