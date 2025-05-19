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



def pathological_partition(labels, num_clients, classes_per_client=2):
    """
    Each client receives only 'classes_per_client' classes.
    """
    n_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    # Assign 'classes_per_client' classes to each client
    class_assignments = []
    for client_idx in range(num_clients):
        # Assign classes so that they are distributed evenly
        if client_idx < n_classes:
            # First round: each class is assigned at least once
            primary_class = client_idx
            remaining_classes = list(range(n_classes))
            remaining_classes.remove(primary_class)
            secondary_classes = np.random.choice(remaining_classes, classes_per_client - 1, replace=False)
            client_classes = np.append([primary_class], secondary_classes)
        else:
            # Subsequent rounds: random assignment
            client_classes = np.random.choice(n_classes, classes_per_client, replace=False)

        class_assignments.append(client_classes)

    # Collect indices for each client
    for client_idx, client_classes in enumerate(class_assignments):
        for class_idx in client_classes:
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)

            # Calculate how many clients have this class
            num_clients_with_class = sum([class_idx in assignment for assignment in class_assignments])
            client_proportion = 1.0 / num_clients_with_class

            # Find the position of this client in the order of clients having this class
            client_position = sum([1 for i in range(client_idx) if class_idx in class_assignments[i]])

            start_idx = int(client_position * client_proportion * len(class_indices))
            end_idx = int((client_position + 1) * client_proportion * len(class_indices))

            client_indices[client_idx].extend(class_indices[start_idx:end_idx])

    return client_indices
