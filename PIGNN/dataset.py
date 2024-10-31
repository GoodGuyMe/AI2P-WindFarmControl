from torch_geometric.data import Dataset

import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, ConcatDataset

class GraphDataset(Dataset):
    def __init__(self, root, preload=True, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.num_samples = len([name for name in os.listdir(self.root) if name != "README.md"])
        self.data = None
        self.graph_paths = None
        self.root = root
        self.preload = preload
        self.load_graph_paths()

    def load_graph_paths(self):
        if self.preload:
            self.data = [torch.load(f"{self.root}/graph_{i}.pt") for i in range(30005, 42000 + 1, 5)]
        else:
            self.graph_paths = [f"{self.root}/graph_{i}.pt" for i in range(30005, 42000 + 1, 5)]

    def len(self):
        return self.num_samples

    def get(self, idx):
        if self.preload:
            return self.data[idx]
        return torch.load(self.graph_paths[idx])


class GraphTemporalDataset(Dataset):
    def __init__(self, root, seq_length, preload=True, transform=None, pre_transform=None):
        super(GraphTemporalDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.seq_length = seq_length
        self.preload = preload

        if preload:
            self.data = [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for start in range(self.len()) for i in range(self.seq_length)]

    def _get_sequence(self, start):
        if self.preload:
            return [self.data[start + i] for i in range(self.seq_length)]
        else:
            return [torch.load(f"{self.root}/graph_{30005 + (start + i) * 5}.pt") for i in range(self.seq_length)]

    def len(self):
        return len([name for name in os.listdir(self.root)]) - 2 * self.seq_length

    def get(self, idx):
        return self._get_sequence(idx), self._get_sequence(idx + self.seq_length)


def get_dataset(dataset_dirs, is_temporal, seq_length):
    datasets = []
    for path in dataset_dirs:
        dataset = GraphTemporalDataset(root=path, seq_length=seq_length) if is_temporal else GraphDataset(
            root=path)
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    print(f"Loaded datasets, {len(dataset)} samples")
    return dataset

def custom_collate_fn(batch):
    # Each batch consists of a list of sequences (each a list of graphs)
    return [Data.from_data_list(seq) for seq in batch]


def create_data_loaders(dataset, batch_size, seq_length):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    collate = custom_collate_fn if seq_length > 1 else None
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)

    return train_loader, val_loader, test_loader
