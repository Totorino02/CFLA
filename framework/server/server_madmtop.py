from typing import List
from framework.client.client_madmtop import ClientMADMTOP
from framework.server.serverbase import Server
import numpy as np
from framework.common.parameter_tree import ParameterTree
import torch


class ServerMADMTOP(Server):
    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.ep1 = args["ep1"]
        self.ep2 = args["ep2"]
        self.device = args["device"]
        self.tree = ParameterTree()
        self.clients : List['ClientMADMTOP'] = []


    def train(self):
        root_id = self.tree.add_root(set([client.client_id for client in self.clients]), self.global_model)
        self.clustered_federated_learning(self.global_model, self.clients, root_id)

    def federated_learning(self, model, clients_subset):
        continuer = True
        delta_params_norms = []
        loss_history = list()
        self.ep1 = 0
        nb_epoch = 0
        while continuer and nb_epoch < 10:
            nb_epoch += 1
            m = len(clients_subset)
            total_samples = 0
            weighted_sum = None
            mean_loss = 0.0
            for client in clients_subset:
                data_size = len(client.train_loader.dataset)
                total_samples += data_size
                updated_params, delta_params, empirical_risk_grad, loss = client.train(model)
                mean_loss += loss
                if weighted_sum is None:
                    weighted_sum = delta_params * data_size
                else:
                    weighted_sum += delta_params * data_size

            # average the update vectors and loss
            mean_loss /= m
            loss_history.append(mean_loss)
            aggregated_delta_params = (weighted_sum / total_samples).detach().clone()


            # update the global model
            offset = 0
            new_state_dict = {}

            #print(aggregated_delta_params.shape)
            for param_name, param in model.named_parameters():
                param_size = param.numel()
                delta = aggregated_delta_params[offset: offset + param_size]
                delta = delta.reshape(param.size())
                new_param = (param + delta).detach().clone()
                new_state_dict[param_name] = new_param
                offset += param_size
            model.load_state_dict(new_state_dict)

            delta_params_norms.append(np.linalg.norm(aggregated_delta_params))

            continuer = np.linalg.norm(aggregated_delta_params) > self.ep1
            self.ep1 = max(delta_params_norms) / 10

        #print("delta params norms:", delta_params_norms)
        return model, loss_history

    def clustered_federated_learning(self, model, clients_subset, root_id):
        """
        Implémentation récursive du Clustered Federated Learning (CFL)
        selon l'algorithme 3 de l'article.
        """

        # Étape 1 : entraînement fédéré initial sur ce sous-ensemble
        model, loss_history = self.federated_learning(model, clients_subset)

        # Evaluation of the model
        acc, test_loss_history = self.evaluate(model, clients_subset)
        g_acc, g_test_loss_history = self.evaluate(model, clients_subset, test_loader=self.test_dataloader)

        # add of information to the parameter tree
        self.tree.add_train_loss_history(root_id, loss_history)
        self.tree.add_test_result(root_id, acc, test_loss_history)
        self.tree.add_global_test_result(root_id, g_acc, g_test_loss_history)
        self.tree.set_model(root_id, model)

        # Étape 2 : calcul des gradients locaux
        gradients = {}
        for client in clients_subset:
            updated_params, delta_params, empirical_risk_grad, loss = client.train(model)
            # grad_norm = np.linalg.norm(empirical_risk_grad)
            #gradients[client.client_id] = grad / grad_norm if grad_norm > 0 else grad
            gradients[client.client_id] = empirical_risk_grad.numpy()

        # Étape 3 : calcul des similarités
        client_ids = [client.client_id for client in clients_subset]
        alpha = np.zeros((len(client_ids), len(client_ids)))
        for i, id_i in enumerate(client_ids):
            for j, id_j in enumerate(client_ids):
                if i != j:
                    alpha[i][j] = np.dot(gradients[id_i], gradients[id_j]) / ( np.linalg.norm(gradients[id_i]) * np.linalg.norm(gradients[id_j]) )

        # Étape 4 : bipartition pour maximiser la dissimilarité
        c1, c2 = self.optimal_bipartition(alpha, clients_subset)
        if len(c2) == 0:
            return c1

        max_score = max(alpha[client_ids.index(c1i.client_id)][client_ids.index(c2j.client_id)] for c1i in c1 for c2j in c2)

        # Étape 5 : vérification du critère de récursion
        gamma = 0.0001
        max_grad = max(np.linalg.norm(gradients[c.client_id]) for c in clients_subset)

        self.ep2 = self.ep1 * 10
        if max_grad >= self.ep2 and np.sqrt((1 - max_score) / 2) > gamma:  # seuil à configurer via args
            print(f"Cluster scindé en 2 (taille : {len(c1)}, {len(c2)})")
            r1_id = self.tree.add_child_cluster(root_id, set([client.client_id for client in c1]), model)
            r2_id = self.tree.add_child_cluster(root_id, set([client.client_id for client in c2]), model)

            m1 = type(model)().to(self.device)
            m2 = type(model)().to(self.device)
            m1.load_state_dict(model.state_dict())
            m2.load_state_dict(model.state_dict())

            r1 = self.clustered_federated_learning(m1, c1, r1_id)
            r2 = self.clustered_federated_learning(m2, c2, r2_id)
            return r1, r2
        else:
            print(f"Cluster convergé (taille : {len(clients_subset)})")
            return clients_subset
            #return {client.client_id: model for client in clients_subset}


    def optimal_bipartition(self, alpha, clients):
        """
        Optimal Bipartition.
        alpha : similarity matrix (shape M x M)
        clients : list of clients (taille M)
        """
        M = len(clients)
        s = np.argsort(-alpha.flatten())  # tri des indices par similarité décroissante
        C = [{i} for i in range(M)]

        device = self.device

        for idx in s:
            i1, i2 = divmod(idx, M)
            ctmp = set()

            # fusion des clusters contenant i1 ou i2
            for c in C:
                if i1 in c or i2 in c:
                    ctmp |= c
            # retrait des anciens clusters fusionnés
            C = [c for c in C if not (i1 in c or i2 in c)]
            C.append(ctmp)

            # si on a exactement deux clusters, on retourne
            if len(C) == 2:
                c1_idx, c2_idx = list(C[0]), list(C[1])
                c1 = [clients[i] for i in c1_idx]
                c2 = [clients[i] for i in c2_idx]
                return c1, c2

        # fallback (non atteint normalement)
        return clients, []

    def evaluate(self, model, clients_subset, test_loader=None, **kwargs):
        model.eval()
        correct_top1 = 0
        total = 0
        total_loss = 0.0
        last_lost = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_history = list()

        if test_loader is None:
            # choose a random client
            m = max(1, len(clients_subset)//2)
            client_ids = np.random.choice(len(clients_subset), m, replace=False)
            test_dataloaders = [clients_subset[client_id].test_loader for client_id in client_ids]
        else:
            test_dataloaders = [test_loader]

        with torch.no_grad():
            for test_dataloader in test_dataloaders:
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    # Top-1
                    _, pred_top1 = outputs.max(dim=1)
                    correct_top1 += (pred_top1 == targets).sum().item()


                    total += targets.size(0)
                    total_loss += loss.item() * targets.size(0)
                    last_lost = loss.item()
                    loss_history.append(loss.item())
        acc_top1 = correct_top1 / total
        avg_loss = total_loss / total

        return acc_top1, loss_history

    def aggregate(self):
        pass

    def get_params(self):
        pass
