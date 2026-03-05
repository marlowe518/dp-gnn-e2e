"""Tests for dataset_readers and input_pipeline modules."""

import pytest
import torch
import numpy as np
from types import SimpleNamespace

from dp_gnn.dataset_readers import DummyDataset, get_dataset
from dp_gnn.input_pipeline import (
    add_reverse_edges,
    add_self_loops,
    compute_masks_for_splits,
    convert_to_pyg_data,
    get_dataset as get_pipeline_dataset,
)


class TestDummyDataset:
    def test_shape(self):
        ds = DummyDataset()
        assert ds.num_nodes() == 9
        assert ds.num_edges() == 9
        assert ds.node_features.shape == (9, 5)
        assert ds.node_labels.shape == (9,)

    def test_splits_cover_all_nodes(self):
        ds = DummyDataset()
        all_nodes = set(ds.train_nodes) | set(ds.validation_nodes) | set(ds.test_nodes)
        assert all_nodes == set(range(ds.num_nodes()))

    def test_get_dataset_dummy(self):
        ds = get_dataset('dummy')
        assert ds.num_nodes() == 9

    def test_get_dataset_unknown_raises(self):
        with pytest.raises(ValueError):
            get_dataset('nonexistent')


class TestAddReverseEdges:
    def test_doubles_edges(self):
        ds = DummyDataset()
        orig_edges = ds.num_edges()
        ds = add_reverse_edges(ds)
        assert ds.num_edges() == 2 * orig_edges


class TestComputeMasks:
    def test_masks_partition(self):
        ds = DummyDataset()
        masks = compute_masks_for_splits(ds)
        combined = masks['train'] | masks['validation'] | masks['test']
        assert np.all(combined)
        assert not np.any(masks['train'] & masks['validation'])


class TestConvertToPyGData:
    def test_output_shape(self):
        ds = DummyDataset()
        data, labels = convert_to_pyg_data(ds)
        assert data.x.shape[0] == ds.num_nodes()
        assert labels.shape[0] == ds.num_nodes()
        assert data.edge_index.shape[1] == ds.num_edges()


class TestAddSelfLoops:
    def test_adds_n_edges(self):
        ds = DummyDataset()
        data, _ = convert_to_pyg_data(ds)
        orig_edges = data.edge_index.size(1)
        data = add_self_loops(data)
        assert data.edge_index.size(1) == orig_edges + ds.num_nodes()


class TestGetPipelineDataset:
    def test_full_pipeline(self):
        config = SimpleNamespace(
            dataset='dummy',
            dataset_path='',
            max_degree=2,
            adjacency_normalization='inverse-degree',
        )
        rng = torch.Generator()
        rng.manual_seed(0)
        data, labels, masks = get_pipeline_dataset(config, rng)

        assert data.x.shape[0] == 9
        assert labels.shape == (9,)
        assert 'train' in masks
        assert 'validation' in masks
        assert 'test' in masks

        # Edge weights should be finite
        assert torch.all(torch.isfinite(data.edge_attr))

    @pytest.mark.parametrize("norm", [None, 'inverse-degree', 'inverse-sqrt-degree'])
    def test_normalizations(self, norm):
        config = SimpleNamespace(
            dataset='dummy',
            dataset_path='',
            max_degree=2,
            adjacency_normalization=norm,
        )
        rng = torch.Generator()
        rng.manual_seed(42)
        data, _, _ = get_pipeline_dataset(config, rng)
        assert torch.all(torch.isfinite(data.edge_attr))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
