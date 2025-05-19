import numpy as np
import itertools

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



def pathological_partition(labels, num_clients, classes_per_client=2, seed=1905):
    """
    Each client receives only 'classes_per_client' classes.
    """
    if seed is not None:
        np.random.seed(seed)

    n_classes = len(np.unique(labels))

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

    return client_indices
