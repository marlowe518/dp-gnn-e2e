"""Dataset readers for DP-GNN.

Provides DummyDataset for testing and OGB dataset readers matching
the reference interface exactly.
"""

import abc
import os

import numpy as np
import pandas as pd


class Dataset(abc.ABC):
    """Abstract base class for datasets."""

    senders: np.ndarray
    receivers: np.ndarray
    node_features: np.ndarray
    node_labels: np.ndarray
    train_nodes: np.ndarray
    validation_nodes: np.ndarray
    test_nodes: np.ndarray

    def num_nodes(self):
        return len(self.node_labels)

    def num_edges(self):
        return len(self.senders)


class DummyDataset(Dataset):
    """A small dummy dataset for testing (mirrors reference)."""

    NUM_DUMMY_TRAINING_SAMPLES: int = 3
    NUM_DUMMY_VALIDATION_SAMPLES: int = 3
    NUM_DUMMY_TEST_SAMPLES: int = 3
    NUM_DUMMY_FEATURES: int = 5
    NUM_DUMMY_CLASSES: int = 3

    def __init__(self):
        n_train = self.NUM_DUMMY_TRAINING_SAMPLES
        n_val = self.NUM_DUMMY_VALIDATION_SAMPLES
        n_test = self.NUM_DUMMY_TEST_SAMPLES
        num_samples = n_train + n_val + n_test

        self.senders = np.arange(num_samples)
        self.receivers = np.roll(np.arange(num_samples), -1)
        self.node_features = np.repeat(
            np.arange(num_samples), self.NUM_DUMMY_FEATURES
        ).reshape(num_samples, self.NUM_DUMMY_FEATURES).astype(np.float32)
        self.node_labels = np.zeros(num_samples, dtype=np.int64)
        self.train_nodes = np.arange(n_train)
        self.validation_nodes = np.arange(n_train, n_train + n_val)
        self.test_nodes = np.arange(n_train + n_val, num_samples)


class OGBTransductiveDataset(Dataset):
    """Reads Open Graph Benchmark (OGB) node-property-prediction datasets.

    Faithfully reproduces the reference implementation's data loading from
    raw CSV files (same file layout as OGB library downloads).
    """

    def __init__(self, dataset_name: str, dataset_path: str):
        super().__init__()
        self.name = dataset_name.replace('-disjoint', '').replace('-', '_')
        base_path = os.path.join(dataset_path, self.name)

        if self.name == 'ogbn_arxiv':
            split_property = 'split/time/'
        elif self.name == 'ogbn_mag':
            split_property = 'split/time/paper/'
        elif self.name == 'ogbn_products':
            split_property = 'split/sales_ranking/'
        elif self.name == 'ogbn_proteins':
            split_property = 'split/species/'
        else:
            raise ValueError(f'Unsupported OGB dataset: {self.name}')

        train_split_file = os.path.join(base_path, split_property, 'train.csv.gz')
        validation_split_file = os.path.join(base_path, split_property, 'valid.csv.gz')
        test_split_file = os.path.join(base_path, split_property, 'test.csv.gz')

        if self.name == 'ogbn_mag':
            node_feature_file = os.path.join(base_path, 'raw/node-feat/paper/node-feat.csv.gz')
            node_label_file = os.path.join(base_path, 'raw/node-label/paper/node-label.csv.gz')
        else:
            node_feature_file = os.path.join(base_path, 'raw/node-feat.csv.gz')
            node_label_file = os.path.join(base_path, 'raw/node-label.csv.gz')

        print(f'Reading node features from {node_feature_file}...')
        self.node_features = pd.read_csv(
            node_feature_file, header=None).values.astype(np.float32)
        print(f'Node features loaded: {self.node_features.shape}')

        print(f'Reading node labels...')
        self.node_labels = pd.read_csv(
            node_label_file, header=None).values.astype(np.int64).squeeze()
        print(f'Node labels loaded: {self.node_labels.shape}')

        if self.name == 'ogbn_mag':
            edge_file = os.path.join(
                base_path, 'raw/relations/paper___cites___paper/edge.csv.gz')
        else:
            edge_file = os.path.join(base_path, 'raw/edge.csv.gz')

        print(f'Reading edges...')
        senders_receivers = pd.read_csv(
            edge_file, header=None).values.T.astype(np.int64)
        self.senders, self.receivers = senders_receivers
        print(f'Edges loaded: {len(self.senders)}')

        print(f'Reading splits...')
        self.train_nodes = pd.read_csv(
            train_split_file, header=None).values.T.astype(np.int64).squeeze()
        self.validation_nodes = pd.read_csv(
            validation_split_file, header=None).values.T.astype(np.int64).squeeze()
        self.test_nodes = pd.read_csv(
            test_split_file, header=None).values.T.astype(np.int64).squeeze()
        print(f'Splits: train={len(self.train_nodes)}, '
              f'val={len(self.validation_nodes)}, test={len(self.test_nodes)}')


class OGBDisjointDataset(OGBTransductiveDataset):
    """A disjoint version of an OGB dataset with no inter-split edges."""

    def __init__(self, dataset_name: str, dataset_path: str):
        super().__init__(dataset_name, dataset_path)
        self.name = dataset_name

        train_split = set(self.train_nodes.flat)
        validation_split = set(self.validation_nodes.flat)
        test_split = set(self.test_nodes.flat)
        splits = [train_split, validation_split, test_split]

        def _compute_split_index(elem):
            elem_index = None
            for index, split in enumerate(splits):
                if elem in split:
                    if elem_index is not None:
                        raise ValueError(f'Node {elem} in multiple splits.')
                    elem_index = index
            if elem_index is None:
                raise ValueError(f'Node {elem} in none of the splits.')
            return elem_index

        print('Computing disjoint edge filter...')
        senders_split = np.vectorize(_compute_split_index)(self.senders)
        receivers_split = np.vectorize(_compute_split_index)(self.receivers)
        in_same_split = (senders_split == receivers_split)

        orig_edges = len(self.senders)
        self.senders = self.senders[in_same_split]
        self.receivers = self.receivers[in_same_split]
        print(f'Disjoint filter: {orig_edges} -> {len(self.senders)} edges')


def get_dataset(dataset_name: str, dataset_path: str = '') -> Dataset:
    """Returns a graph dataset by name."""
    if dataset_name == 'dummy':
        return DummyDataset()

    if dataset_name.startswith('ogb'):
        if dataset_name.endswith('disjoint'):
            return OGBDisjointDataset(dataset_name, dataset_path)
        return OGBTransductiveDataset(dataset_name, dataset_path)

    raise ValueError(f'Unsupported dataset: {dataset_name}.')
