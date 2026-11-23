"""Save completed optimizer updates and retry validation without retraining."""
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random

import numpy as np
import torch


RECOVERY_INTERVAL_SECONDS = 300


def atomic_json(path, value):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(destination)


def atomic_torch(path, value):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(destination)


def rng_state(device):
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}


def restore_rng(state, device):
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda_rng'])


@dataclass
class EpochProgress:
    train_count: int = 0
    sequence_loss_total: float = 0.
    auxiliary_loss_total: float = 0.
    distillation_loss_total: float = 0.
    token_count: int = 0
    annotated_count: int = 0
    skipped_updates: int = 0
    batch_index: int = -1
    epoch_updates: int = 0
    completed_groups: int = 0
    train_seconds: float = 0.
    adaptive_loss_total: float = 0.
    adaptive_episodes: int = 0
    adaptive_source_draws: Counter = field(default_factory=Counter)
    source_sequence_mass: Counter = field(default_factory=Counter)
    source_target_tokens: Counter = field(default_factory=Counter)
    epoch_sampling_draws: Counter = field(default_factory=Counter)

    def to_dict(self):
        return {key: dict(value) if isinstance(value, Counter) else value
                for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, value):
        result = cls(**value)
        for name in ('source_sequence_mass', 'source_target_tokens', 'epoch_sampling_draws','adaptive_source_draws'):
            setattr(result, name, Counter(getattr(result, name)))
        return result

    def loss(self, teacher_weight):
        sequence = ((1 - teacher_weight) * self.sequence_loss_total
                    + teacher_weight * self.distillation_loss_total) / max(self.token_count, 1)
        return sequence + self.auxiliary_loss_total / max(self.annotated_count, 1)
