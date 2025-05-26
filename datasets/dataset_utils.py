import numpy as np
import itertools
from collections import defaultdict
import torch


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

def make_non_iid(non_iid_type, labels, num_clients, **kwargs):
    if non_iid_type == "dirichlet":
        client_indices = dirichlet_partition(labels, num_clients, **kwargs)
    elif non_iid_type == "pathological":
        client_indices = pathological_partition(labels, num_clients, **kwargs)
    else:
        raise ValueError("Unknown partition type")
    return client_indices


