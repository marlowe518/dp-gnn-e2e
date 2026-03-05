"""Tests for ogbn-arxiv dataset loading and pipeline."""

import pytest
import torch
from types import SimpleNamespace

from dp_gnn.dataset_readers import get_dataset, OGBTransductiveDataset
from dp_gnn import input_pipeline


class TestOGBArxivLoading:
    def test_basic_loading(self):
        ds = get_dataset('ogbn-arxiv', 'datasets/')
        assert ds.num_nodes() == 169343
        assert ds.node_features.shape == (169343, 128)
        assert len(set(ds.node_labels)) == 40
        assert len(ds.train_nodes) == 90941
        assert len(ds.validation_nodes) == 29799
        assert len(ds.test_nodes) == 48603

    def test_full_pipeline_mlp(self):
        config = SimpleNamespace(
            dataset='ogbn-arxiv',
            dataset_path='datasets/',
            max_degree=1,
            adjacency_normalization='inverse-degree',
        )
        rng = torch.Generator()
        rng.manual_seed(0)
        data, labels, masks = input_pipeline.get_dataset(config, rng)

        assert data.x.shape[0] == 169343
        assert data.x.shape[1] == 128
        assert labels.shape == (169343,)
        assert masks['train'].sum().item() == 90941
        assert torch.all(torch.isfinite(data.edge_attr))

    def test_full_pipeline_gcn(self):
        config = SimpleNamespace(
            dataset='ogbn-arxiv',
            dataset_path='datasets/',
            max_degree=30,
            adjacency_normalization='inverse-degree',
        )
        rng = torch.Generator()
        rng.manual_seed(0)
        data, labels, masks = input_pipeline.get_dataset(config, rng)

        assert data.x.shape[0] == 169343
        assert torch.all(torch.isfinite(data.edge_attr))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
