from torch.utils.data import Dataset
from torchvision import transforms, datasets
from datasets.dataset_utils import dirichlet_partition, pathological_partition
import numpy as np

class EMNISTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.labels = np.unique(data.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_emnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    return train_dataset, test_dataset

def make_non_iid(non_iid_type, labels, num_clients, **kwargs):
    if non_iid_type == "dirichlet":
        client_indices = dirichlet_partition(labels, num_clients, **kwargs)
    elif non_iid_type == "pathological":
        client_indices = pathological_partition(labels, num_clients, **kwargs)
    else:
        raise ValueError("Unknown partition type")
    return client_indices
