import numpy as np
import itertools
from collections import defaultdict
import torch


def split_labels_structured(labels, num_clients, classes_per_client):
    """
    Splits dataset indices among clients in a structured non-IID way:
    - Classes are divided into equal-sized blocks (clusters).
    - Each cluster is assigned to a group of clients.
    - Clients within the same cluster share the same set of classes (no class overlap between clusters).

    Args:
        labels (np.ndarray): 1D array of dataset labels.
        num_clients (int): Total number of clients (must be divisible by number of clusters).
        classes_per_client (int): Number of classes assigned to each client (must divide total number of classes).

    Returns:
        dict: Mapping from client_id to list of sample indices.
    """
    all_classes = np.unique(labels)
    num_classes = len(all_classes)

    assert num_classes % classes_per_client == 0, "Number of classes must be divisible by classes_per_client"

    num_clusters = num_classes // classes_per_client
    assert num_clients % num_clusters == 0, "Number of clients must be divisible by number of clusters"

    clients_per_cluster = num_clients // num_clusters

    # Map each class to its sample indices
    class_to_indices = {cls: np.where(labels == cls)[0].tolist() for cls in all_classes}

    # Optional: sort for reproducibility
    sorted_classes = sorted(all_classes)

    client_dict = defaultdict(list)
    client_id = 0

    for cluster_id in range(num_clusters):
        # Select classes for this cluster
        cluster_classes = sorted_classes[cluster_id * classes_per_client: (cluster_id + 1) * classes_per_client]

        # Collect all indices belonging to these classes
        cluster_indices = []
        for cls in cluster_classes:
            cluster_indices.extend(class_to_indices[cls])

        # Shuffle to mix samples before splitting
        np.random.shuffle(cluster_indices)

        # Split data evenly among clients in this cluster
        split_size = len(cluster_indices) // clients_per_cluster
        for i in range(clients_per_cluster):
            start = i * split_size
            end = (i + 1) * split_size if i != clients_per_cluster - 1 else len(cluster_indices)
            client_dict[client_id] = cluster_indices[start:end]
            client_id += 1

    return client_dict


def dirichlet_partition(labels, num_clients, alpha=0.1):
    """
        Partition data according to a Dirichlet distribution
        alpha: concentration parameter (the smaller the alpha, the greater the heterogeneity)
    """
    # dictionary to store indices which we will attribute to each client
    clients_data_indices = defaultdict(list)

    min_size = 0
    K = torch.unique(labels).numel()
    N = len(labels)

    # guarantee that each client must have at least one batch of data for testing.
    least_samples = 100

    idx_batch = [[] for _ in range(num_clients)]
    try_cnt = 1
    while min_size < least_samples and try_cnt <= 100: # we try 100 times to get the minimum sample size for each client.
        # if try_cnt > 1:
            # print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        try_cnt += 1

    for j in range(num_clients):
        clients_data_indices[j] = idx_batch[j]

    return clients_data_indices


def pathological_partition(labels, num_clients, classes_per_client=2, seed=None):
    """
    Each client receives only 'classes_per_client' classes.
    """
    if seed is not None:
        np.random.seed(seed)

    n_classes =  torch.unique(labels).numel()

    # Generate all possible permutations of class assignments (order matters)
    all_arrangements = list(itertools.permutations(range(n_classes), classes_per_client))

    # If the number of clients exceeds the number of unique arrangements,
    # reuse arrangements with wraparound and shuffle for diversity
    if num_clients > len(all_arrangements):
        selected_arrangements = [all_arrangements[i % len(all_arrangements)] for i in range(num_clients)]
        np.random.shuffle(selected_arrangements)  # Shuffle to avoid repeated patterns
    else:
        indices = np.random.choice(len(all_arrangements), num_clients, replace=False)
        selected_arrangements = [all_arrangements[i] for i in indices]

    client_indices = [ list(client_label) for client_label in selected_arrangements]

    # Create a dictionary to store indices for each target class
    class_indices = defaultdict(list)
    for idx, target in enumerate(labels):
        class_indices[int(target)].append(idx)
    
    # dictionary to store indices which we will attribute to each client
    clients_data_indices = defaultdict(list)

    # For each class, we distribute its indices among the clients who own that class
    for target, indices in class_indices.items():
        # Find clients who have this class in their targets
        clients_with_target = [cid for cid, targets in enumerate(client_indices) if target in targets]
        if not clients_with_target:
            continue
        # Distribute indices evenly among these clients
        for i, idx in enumerate(indices):
            client_id = clients_with_target[i % len(clients_with_target)]
            clients_data_indices[client_id].append(idx)

    return clients_data_indices

def patho(labels, num_clients, classes_per_client=2, balance=True, least_samples=100):

    # dictionary to store indices which we will attribute to each client
    clients_data_indices = defaultdict(np.ndarray)

    idxs = np.array(range(len(labels)))
    num_classes = len(np.unique(labels))
    idx_for_each_class = []

    for i in range(num_classes):
        idx_for_each_class.append(idxs[labels == i])

    class_num_per_client = [classes_per_client for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
        if len(selected_clients) == 0:
            break
        selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * classes_per_client))]

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        if balance:
            num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
        else:
            num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                            num_selected_clients - 1).tolist()
        num_samples.append(num_all_samples - sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in clients_data_indices.keys():
                clients_data_indices[client] = idx_for_each_class[i][idx:idx + num_sample]
            else:
                clients_data_indices[client] = np.append(clients_data_indices[client], idx_for_each_class[i][idx:idx + num_sample],
                                                axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1
    return clients_data_indices

def make_non_iid(non_iid_type, labels, num_clients, **kwargs):
    if non_iid_type == "dirichlet":
        client_indices = dirichlet_partition(labels, num_clients, **kwargs)
    elif non_iid_type == "pathological":
        client_indices = pathological_partition(labels, num_clients, **kwargs)
    else:
        raise ValueError("Unknown partition type")
    return client_indices


