# utils.py
# Utility functions: IO, metrics, DDP helpers

import os
import json
import numpy as np
import pandas as pd
from ast import literal_eval

import torch
import torch.distributed as dist

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


# -----------------------------
# IO Helpers
# -----------------------------
def read_jsonl(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


def get_checkpoint_path(model_name: str):
    return f"saved_models/{model_name}"


def load_annotated_data(args, path_csv):
    df = pd.read_csv(
        path_csv,
        converters={col: literal_eval for col in ["exercises_logs", "is_corrects", "KCs", "Description"]},
    )

    train_df = df[0:int(0.8 * len(df))]
    test_df = df[int(0.8 * len(df)):]
    return (
        train_df[:int(.8 * len(train_df))],
        train_df[int(.8 * len(train_df)):],
        test_df
    )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(labels: np.ndarray, preds: np.ndarray):
    hard = (preds >= 0.5).astype(np.int32)
    acc = accuracy_score(labels, hard)

    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard, average="binary", zero_division=0)
    return acc * 100.0, auc * 100.0, prec * 100.0, f1 * 100.0


# -----------------------------
# DDP Helpers
# -----------------------------
def is_torchrun():
    return "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def gather_1d_tensor(t: torch.Tensor) -> torch.Tensor:
    """All-gather variable-length 1D tensors across ranks; returns concatenated tensor."""
    t = t.contiguous()
    world_size = dist.get_world_size()

    sizes = torch.tensor([t.numel()], device=t.device, dtype=torch.long)
    sizes_list = [torch.zeros_like(sizes) for _ in range(world_size)]
    dist.all_gather(sizes_list, sizes)
    sizes_list = [int(x.item()) for x in sizes_list]
    max_size = max(sizes_list)

    if t.numel() < max_size:
        pad = torch.zeros(max_size - t.numel(), device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    gather_list = [torch.zeros(max_size, device=t.device, dtype=t.dtype) for _ in range(world_size)]
    dist.all_gather(gather_list, t_pad)

    out = []
    for g, sz in zip(gather_list, sizes_list):
        out.append(g[:sz])
    return torch.cat(out, dim=0)
