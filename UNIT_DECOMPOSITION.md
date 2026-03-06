# Unit Decomposition: Pre-train + DP Finetuning Framework

## Unit 1: Checkpoint Utilities (`checkpoint_utils.py`)

**Purpose:** Save and load model checkpoints with metadata

**Input:**
- Model (nn.Module)
- Path (str)
- Metadata dict (optional)

**Output:**
- Saved checkpoint file (.pt)
- Loaded state_dict + metadata

**Dependencies:** None (pure PyTorch)

**Success Criteria:**
- Can save and load MLP state_dict
- Can save and load GCN state_dict
- Metadata preserved round-trip

**Test Plan:** `tests/test_checkpoint_utils.py`
- Test save/load MLP
- Test save/load GCN
- Test metadata round-trip

---

## Unit 2: Parameter Transfer (`transfer.py`)

**Purpose:** Map MLP parameters to GCN parameters

**Input:**
- MLP state_dict
- GCN model
- Transfer strategy: 'encoder_only', 'classifier_only', 'encoder_classifier', 'full'

**Output:**
- GCN with loaded parameters

**Dependencies:** checkpoint_utils (for testing), models

**Success Criteria:**
- Encoder layers transfer correctly
- Classifier layer transfers correctly
- Shape mismatches raise clear errors
- Missing layers handled gracefully

**Test Plan:** `tests/test_transfer.py`
- Test each transfer strategy
- Test shape validation
- Test partial transfer (when MLP deeper than GCN)

---

## Unit 3: MLP Pretraining (`pretrain.py`)

**Purpose:** Standard (non-DP) MLP training on node features

**Input:**
- Config with pretraining settings
- Optional: validation for early stopping

**Output:**
- Trained MLP model
- Training history (loss, accuracy)
- Saved checkpoint

**Dependencies:** models, dataset_readers, input_pipeline, checkpoint_utils

**Success Criteria:**
- Training loss decreases
- Validation accuracy increases
- Model checkpoint saved
- Can train on full graph (not just train split)

**Test Plan:** `tests/test_pretrain.py`
- Test training loop on dummy data
- Test checkpoint saving
- Test loss decrease over iterations

---

## Unit 4: Pretraining Configs (`configs/pretrain_mlp.py`)

**Purpose:** Configuration for MLP pretraining

**Input:** None (returns SimpleNamespace)

**Output:** Config object with:
- Architecture: latent_size, num_layers, activation
- Training: epochs, lr, batch_size, optimizer
- Data: dataset, use_full_graph
- Checkpoint: save_path

**Dependencies:** None

**Success Criteria:**
- Config loads without error
- All required fields present
- Can override fields

**Test Plan:** `tests/test_configs.py`
- Test config creation
- Test field override

---

## Unit 5: Extended Training Pipeline (`train_with_pretrain.py`)

**Purpose:** Integrate pretrained initialization into DP finetuning

**Input:**
- Config with pretrain_checkpoint_path
- Optional: freeze_strategy

**Output:**
- Trained GCN/DP-GCN model
- Metrics log

**Dependencies:** train.py (existing), transfer.py, checkpoint_utils

**Success Criteria:**
- Loads pretrained MLP if checkpoint provided
- Applies transfer strategy correctly
- Supports layer freezing
- Maintains existing DP training behavior

**Test Plan:** `tests/test_train_with_pretrain.py`
- Test pretrained initialization
- Test freeze strategies
- Test equivalence to baseline when no checkpoint

---

## Unit 6: Experiment Scripts

### 6.1 `scripts/pretrain_mlp_arxiv.py`
**Purpose:** Run MLP pretraining on ogbn-arxiv

**Input:** Command line args or config
**Output:** Trained MLP checkpoint

### 6.2 `scripts/finetune_dpgcn_with_pretrain.py`
**Purpose:** DP-GCN finetuning from pretrained MLP

**Input:** Pretrain checkpoint path, finetune config
**Output:** Trained DP-GCN, metrics

### 6.3 `scripts/run_pretrain_finetune_pipeline.py`
**Purpose:** End-to-end pretrain + finetune

**Input:** Experiment config
**Output:** Both checkpoints, comparison metrics

---

## Implementation Order

1. Unit 1 (Checkpoint Utils) - Foundation
2. Unit 4 (Pretrain Configs) - Simple, no deps
3. Unit 2 (Transfer) - Depends on Unit 1
4. Unit 3 (Pretraining) - Depends on Units 1, 4
5. Unit 5 (Extended Training) - Depends on Units 2, 3
6. Unit 6 (Scripts) - Integration layer

## Integration Points

```
models.py (existing)
    ↓
checkpoint_utils.py (Unit 1)
    ↓
transfer.py (Unit 2) ←→ configs/pretrain_mlp.py (Unit 4)
    ↓
pretrain.py (Unit 3)
    ↓
train_with_pretrain.py (Unit 5)
    ↓
scripts/ (Unit 6)
```
