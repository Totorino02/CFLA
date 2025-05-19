import numpy as np


def dirichlet_partition(labels, num_clients, alpha=0.5):
    """
    Partition data according to a Dirichlet distribution
    alpha: concentration parameter (the smaller the alpha, the greater the heterogeneity)
    """
    n_classes = len(np.unique(labels))
    client_label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

    client_indices = [[] for _ in range(num_clients)]
    for class_idx in range(n_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)

        # Distribution of samples according to the Dirichlet distribution
        proportions = client_label_distribution[class_idx]
        proportions = proportions / proportions.sum()  # Normalization
        cumulative_proportions = np.cumsum(proportions)

        start_idx = 0
        for client_idx, end_proportion in enumerate(cumulative_proportions):
            end_idx = int(end_proportion * len(class_indices))
            client_indices[client_idx].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    return client_indices



