from typing import Dict, List, Set, Any
import numpy as np
import torch

class TreeNode:
    def __init__(self, cluster_id: int, clients: Set[int], model: torch.nn.Module):
        self.id = cluster_id
        self.clients = clients  # Set of client indices in this cluster
        self.model = model  # Stationary solution for this cluster
        self.train_loss_history = list()
        self.test_loss_history = list()
        self.global_test_loss_history = list()
        self.global_accuracy = 0.0
        self.accuracy = 0.0
        self.children: List['TreeNode'] = []
        self.parent: 'TreeNode' = None
        self.delta_to_children: Dict[int, List[np.ndarray]] = {}  # child_id -> list of deltas

class ParameterTree:
    def __init__(self):
        self.nodes: Dict[int, TreeNode] = {}
        self.root: TreeNode = None
        self.next_cluster_id = 0

    def add_root(self, clients: Set[int], model: torch.nn.Module) -> int:
        root_id = self.next_cluster_id
        self.next_cluster_id += 1
        root = TreeNode(root_id, clients, model)
        self.nodes[root_id] = root
        self.root = root
        return root_id

    def set_model(self,node_id,  model: torch.nn.Module):
        self.nodes[node_id].model = model

    def add_train_loss_history(self, node_id, train_loss_history: List[float]):
        self.nodes[node_id].train_loss_history = train_loss_history

    def add_test_result(self, node_id, accuracy: float, test_loss_history: List[float]):
        self.nodes[node_id].accuracy = accuracy
        self.nodes[node_id].test_loss_history = test_loss_history

    def add_global_test_result(self, node_id, g_accuracy: float, global_test_loss_history: List[float]):
        self.nodes[node_id].global_test_loss_history = global_test_loss_history
        self.nodes[node_id].global_accuracy = g_accuracy

    def add_child_cluster(self, parent_id: int, child_clients: Set[int], model: torch.nn.Module) -> int:
        child_id = self.next_cluster_id
        self.next_cluster_id += 1

        parent_node = self.nodes[parent_id]
        child_node = TreeNode(child_id, child_clients, model)
        child_node.parent = parent_node
        parent_node.children.append(child_node)

        # Compute deltas: update steps from parent weights to child clients
        """deltas = []
        for client_id in child_clients:
            delta = sgd_update_fn(parent_node.theta_star, client_id)
            deltas.append(delta)
        parent_node.delta_to_children[child_id] = deltas"""

        self.nodes[child_id] = child_node
        return child_id

    def get_node(self, cluster_id: int) -> TreeNode:
        return self.nodes[cluster_id]

    def visualize_tree(self, filename):
        def _write_node(node: TreeNode, depth: int, file):
            indent = "  " * depth
            file.write(f"{indent}- Cluster {node.id} | Clients: {sorted(node.clients)} | Acc: {node.accuracy} | G_acc: {node.global_accuracy}\n")
            for child in node.children:
                _write_node(child, depth + 1, file)

        with open(filename, 'w') as file:
            if self.root is not None:
                file.write("Parameter Tree:\n")
                _write_node(self.root, 0, file)
            else:
                file.write("The tree is empty.\n")


