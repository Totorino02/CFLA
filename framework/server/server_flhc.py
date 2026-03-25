from collections import defaultdict
from datetime import datetime
import os
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
        self.output_dir = args["output_dir"]
        self.seed = args.get("seed", 0)
        self.rng = np.random.default_rng(self.seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def pre_learning(self):
        for epoch in range(self.initial_rounds):
            self.global_model, loss = self.federated_learning(self.global_model, self.clients, round=-(self.initial_rounds - epoch))

    def federated_learning(self, model, clients_subset, **kwargs):
        """
        This method performs federated learning on a subset of clients and returns the updated model
        and the mean loss of the local training.
        :param model: Model to be updated
        :param clients_subset: The subset of clients to perform federated learning on
        :return: An updated model, mean loss
        """
        # selects clients
        m = max(1, int(self.fraction * len(clients_subset)))
        selected_clients = self.rng.choice(clients_subset, m, replace=False)

        total_samples = 0
        weighted_sum = None
        mean_loss = 0.0
        for client in selected_clients:
            data_size = len(client.train_loader.dataset)
            updated_params, update_vector, loss = client.train(model, round=kwargs.get("round", 0))
            mean_loss += loss
            total_samples += data_size
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
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric=self.clustering_metric,
            linkage='average'
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
        # Collect per-round acc/loss across all clients (all clusters) for server_metrics
        round_accs = defaultdict(list)
        round_losses = defaultdict(list)
        print("Specialized models training...")
        personalization_start_time = time.time()
        for label, cluster_clients in self.clusters.items():
            cluster_model = type(self.global_model)().to(self.device)
            cluster_model.load_state_dict(self.global_model.state_dict())
            print(f"Cluster {int(label)+1} training...")
            for _round in tqdm(range(self.cluster_rounds), unit="round", colour="green"):
                cluster_model, mean_loss = self.federated_learning(cluster_model, cluster_clients, round=_round)
                training_loss_history[int(label)].append(mean_loss)
                g_acc_top1, g_acc_topk, g_test_loss = self.evaluate(cluster_model, self.test_dataloader, k=1, return_loss=True)
                global_test_accuracy_history[int(label)].append(g_acc_top1)
                global_test_loss_history[int(label)].append(g_test_loss)
                # Evaluate all clients in this cluster so every CSV has one row per round
                for c in cluster_clients:
                    acc_top1, _, test_loss = self.evaluate(cluster_model, c.test_loader, k=1, return_loss=True)
                    with open(os.path.join(self.output_dir, f"client_{c.client_id}", "metrics.csv"), "a") as f:
                        f.write(f"{_round},{test_loss},{acc_top1},{acc_top1},0,0\n")
                    round_accs[_round].append(acc_top1)
                    round_losses[_round].append(test_loss)
                test_loss_history[int(label)].append(test_loss)
                accuracy_history[int(label)].append(acc_top1)
            specialized_models[label] = cluster_model
        self.specialized_models = specialized_models

        # Write global server_metrics: mean/std across all clients at each round
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "a") as f:
            for r in range(self.cluster_rounds):
                f.write(
                    f"{r},{np.mean(round_accs[r]):.6f},"
                    f"{np.std(round_accs[r]):.6f},"
                    f"{np.mean(round_losses[r]):.6f}\n"
                )
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
            return acc_top1, acc_topk, avg_loss
        else:
            return acc_top1, acc_topk

    def aggregate(self):
        pass

    def set_clients(self, clients: list[ClientFLHC]):
        self.clients = clients
        for client in clients:
            client.output_dir = self.output_dir

        # Initialize server metrics CSV
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    def get_params(self):
        return self.global_model.state_dict()

