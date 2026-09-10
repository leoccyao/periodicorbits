"""Utilities for loading and summarizing orbit trainers and branches."""

import torch
import numpy as np
import pandas as pd
from orbitlib import *
from doublepend import *

import matplotlib.pyplot as plt
from typing import Iterable

import os
import json

SYSTEM = Double_Pendulum_Orbit()

torch.set_default_dtype(torch.float64)

def get_match_dir_names(curve_data_dir):
    """Return numbered branch-result directory names in propagation order."""
    return sorted(
        [a.name for a in os.scandir(curve_data_dir) if ' [' in a.name], 
        key=lambda name: int(name.split(' ', maxsplit=1)[0])
    )


def trainer_index(curve_data_dir, index) -> Training_Progress:
    """Load a trainer by its position in propagation order."""
    return load_trainer(curve_data_dir, get_match_dir_names(curve_data_dir)[index])


def load_trainer(curve_data_dir, match):
    """Load a serialized trainer from one branch-result directory."""
    return torch.load(f'{curve_data_dir}/{match}/var-model.pt')


def stream_paths_trainers(curve_data_dir, limit=None) -> Generator[tuple[str, Trainer], None, None]:
    """Yield result-directory names and trainers without retaining the branch."""

    match_dir_names = get_match_dir_names(curve_data_dir)[:limit]
    for match in match_dir_names:
        yield match, load_trainer(curve_data_dir, match)


def stream_last_checkpoints(curve_data_dir, limit=None) -> Generator[Training_Progress, None, None]:
    """Yield the final checkpoint from each trainer in propagation order."""

    for match, trainer in stream_paths_trainers(curve_data_dir, limit=limit):
        yield trainer.checkpoints[-1]


def streaming_apply_trainers(curve_data_dir, func):
    """Apply ``func`` to each streamed trainer and return the results."""

    return [
        func(trainer)
        for match, trainer in stream_paths_trainers(curve_data_dir)
    ]


def degree_squeeze_angles(initial_condition):
    """Convert a tensor-valued state from radians to a squeezed degree array."""
    return 180 / np.pi * initial_condition.detach().numpy().squeeze()


def get_train_info(last_checkpoints: Iterable[Training_Progress]):
    """Build a table of loss, period, energy, and initial-state diagnostics."""
    return pd.DataFrame([
        [c.train_loss.item(), c.val_loss, c.T.item(), SYSTEM.energy(c.initial_condition).item(), *degree_squeeze_angles(c.initial_condition)]
        for c in last_checkpoints
    ], columns=['Train_Loss', 'Val_Loss', 'T', 'Energy', 'Theta1', 'Theta2', 'Theta1Dot', 'Theta2Dot'])


def get_trainers_checkpoints(curve_data_dir):
    """Load all trainers and final checkpoints into memory."""
    trainers: list[Trainer] = [
        trainer for match, trainer in stream_paths_trainers(curve_data_dir)
    ]
    last_checkpoints = [t.checkpoints[-1] for t in trainers]

    return trainers, last_checkpoints


def get_trainers_dict(curve_data_dir) -> dict[str, Trainer]:
    """Load all trainers into a dictionary keyed by result-directory name."""
    trainers = {
        match: trainer for match, trainer in stream_paths_trainers(curve_data_dir)
    }
    return trainers
