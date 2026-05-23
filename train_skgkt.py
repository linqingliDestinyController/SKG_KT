# train_skgkt.py
# SKG-KT training script supporting both single-GPU and multi-GPU (DDP)
#
# Single GPU:
#   python train_skgkt.py train
#
# Multi-GPU (DDP):
#   torchrun --nproc_per_node=8 train_skgkt.py train

import os
import json
import argparse
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import transformers

from utils import (
    read_jsonl, get_checkpoint_path, load_annotated_data,
    compute_metrics, is_torchrun, ddp_setup, ddp_cleanup, gather_1d_tensor,
)
from data import SKGKTDatasetPacked, SKGKTCollatorPacked
from model import get_skgkt_model, get_true_false_tokens, get_skgkt_loss_packed


# -----------------------------
# Args
# -----------------------------
def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    train_p = subparsers.add_parser("train", help="Train SKG-KT")
    train_p.add_argument("--dataset", type=str, choices=["xes3g5m", "moocradar", "Eedi"], default="Eedi")
    train_p.add_argument("--epochs", type=int, default=5)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--wd", type=float, default=1e-2)
    train_p.add_argument("--gc", type=float, default=1.0)
    train_p.add_argument("--grad_accum_steps", type=int, default=32)
    train_p.add_argument("--batch_size", type=int, default=2)
    train_p.add_argument("--r", type=int, default=8)
    train_p.add_argument("--lora_alpha", type=int, default=8)
    train_p.add_argument("--optim", type=str, choices=["adamw", "adafactor"], default="adamw")
    train_p.add_argument("--pt_model_name", type=str, default=None)
    train_p.add_argument("--model_name", type=str, default="skgkt")
    train_p.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    train_p.add_argument("--agg", type=str, choices=["prod", "mean-ar", "mean-geo"], default="mean-ar")
    train_p.add_argument("--inc_first_label", action="store_true")
    train_p.add_argument("--prompt_inc_labels", action="store_true")
    train_p.add_argument("--debug", action="store_true")
    train_p.add_argument("--seed", type=int, default=221)
    train_p.add_argument("--num_workers", type=int, default=4)
    train_p.add_argument("--quantize", action="store_true", help="Not recommended on H100.")
    train_p.add_argument("--max_length", type=int, default=4096)

    return parser


# -----------------------------
# Eval (single-GPU)
# -----------------------------
def eval_on_loader_single(model, dataloader, true_token, false_token, args):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, corr_probs = get_skgkt_loss_packed(model, batch, true_token, false_token, args)
            total_loss += float(loss.item())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_preds.extend(corr_probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc, auc, prec, f1 = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, auc, acc, f1


# -----------------------------
# Eval (DDP)
# -----------------------------
def eval_on_loader_ddp(model, dataloader, true_token, false_token, args, local_rank):
    model.eval()
    total_loss = 0.0
    labels_local = []
    preds_local = []

    with torch.no_grad():
        for batch in dataloader:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, corr_probs = get_skgkt_loss_packed(model, batch, true_token, false_token, args)
            total_loss += float(loss.item())
            labels_local.append(batch["labels"].detach().float().to(torch.device("cuda", local_rank)))
            preds_local.append(corr_probs.detach().float().to(torch.device("cuda", local_rank)))

    loss_tensor = torch.tensor([total_loss], device=torch.device("cuda", local_rank), dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    labels_local = torch.cat(labels_local, dim=0)
    preds_local = torch.cat(preds_local, dim=0)

    labels_all = gather_1d_tensor(labels_local).cpu().numpy()
    preds_all = gather_1d_tensor(preds_local).cpu().numpy()

    world_size = dist.get_world_size()
    avg_loss = (loss_tensor.item() / world_size) / max(1, len(dataloader))

    acc, auc, prec, f1 = compute_metrics(labels_all, preds_all)
    return avg_loss, auc, acc, f1


# -----------------------------
# Test (standalone, no DDP needed)
# -----------------------------
def test_skgkt(args, fold, texts, kgs, path_data, local_rank=0):
    model_base = args.model_name + (f"_{fold}" if fold else "")
    model, tokenizer = get_skgkt_model(
        args.base_model,
        test=True,
        local_rank=local_rank,
        model_name=model_base,
        quantize=args.quantize,
    )
    model.eval()

    _, val_df, test_df = load_annotated_data(args, path_data)
    if args.debug:
        test_df = test_df[:50].reset_index(drop=True)

    test_dataset = SKGKTDatasetPacked(test_df, texts, kgs, tokenizer, args, skip_first_turn=not args.inc_first_label)
    collator = SKGKTCollatorPacked(tokenizer, max_length=args.max_length)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=getattr(args, "num_workers", 4),
        pin_memory=True, persistent_workers=(getattr(args, "num_workers", 4) > 0),
    )

    true_token, false_token = get_true_false_tokens(tokenizer)

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in tqdm(test_loader, desc="[TEST]"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, corr_probs = get_skgkt_loss_packed(model, batch, true_token, false_token, args)
        total_loss += float(loss.item())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())
        all_preds.extend(corr_probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(test_loader))
    acc, auc, prec, f1 = compute_metrics(np.array(all_labels), np.array(all_preds))

    print(f"\n[TEST RESULT] Loss={avg_loss:.4f} | ACC={acc:.2f} AUC={auc:.2f} Prec={prec:.2f} F1={f1:.2f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics_{model_base}.txt", "w", encoding="utf-8") as f:
        f.write(f"Loss: {avg_loss:.4f}\nACC: {acc:.2f}, AUC: {auc:.2f}, Prec: {prec:.2f}, F1: {f1:.2f}\n")
    return np.array([avg_loss, acc, auc, prec, f1], dtype=np.float32)


# -----------------------------
# Train (single-GPU)
# -----------------------------
def train_skgkt_single(args, fold, texts, kgs, path_data):
    local_rank = 0
    device = torch.device("cuda", local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model, tokenizer = get_skgkt_model(
        args.base_model, test=False, local_rank=local_rank,
        pt_model_name=args.pt_model_name, r=args.r,
        lora_alpha=args.lora_alpha, quantize=args.quantize,
    )

    train_df, val_df, _ = load_annotated_data(args, path_data)
    train_dataset = SKGKTDatasetPacked(train_df, texts, kgs, tokenizer, args)
    val_dataset = SKGKTDatasetPacked(val_df, texts, kgs, tokenizer, args)
    collator = SKGKTCollatorPacked(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )

    true_token, false_token = get_true_false_tokens(tokenizer)

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.wd, relative_step=False)

    best_key = None
    best_val_loss = None
    model_base = args.model_name + (f"_{fold}" if fold else "")
    history_path = f"results/history_{model_base}.jsonl"

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training E{epoch+1}")):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = get_skgkt_loss_packed(model, batch, true_token, false_token, args)

            total_train_loss += float(loss.item())
            (loss / args.grad_accum_steps).backward()

            do_step = ((step + 1) % args.grad_accum_steps == 0) or (step == len(train_loader) - 1)
            if do_step:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss, val_auc, val_acc, val_f1 = eval_on_loader_single(
            model, val_loader, true_token, false_token, args
        )

        # Save every epoch
        epoch_name = f"{model_base}_epoch{epoch+1:02d}"
        epoch_dir = get_checkpoint_path(epoch_name)
        model.save_pretrained(epoch_dir)
        print(f"[Checkpoint] Saved epoch {epoch+1} -> {epoch_dir}")

        # Track best
        curr_key = (val_auc, val_acc, val_f1)
        is_best = best_key is None or curr_key > best_key

        print(f"Epoch {epoch+1}: ValLoss={val_loss:.4f} | ValAUC={val_auc:.2f} ValACC={val_acc:.2f} ValF1={val_f1:.2f}")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "val_loss": float(val_loss),
                "val_auc": float(val_auc), "val_acc": float(val_acc), "val_f1": float(val_f1),
                "epoch_dir": epoch_dir,
            }, ensure_ascii=False) + "\n")

        if is_best:
            best_key = curr_key
            best_val_loss = val_loss
            best_dir = get_checkpoint_path(model_base)
            model.save_pretrained(best_dir)
            print(f"[BEST] Saved best -> {best_dir} | best_key={best_key} | best_val_loss={best_val_loss:.4f}")

    print("\nTraining finished. Best key:", best_key)
    print("[TEST] Running test with BEST checkpoint...")
    test_skgkt(args, fold, texts, kgs, path_data, local_rank=local_rank)


# -----------------------------
# Train (DDP)
# -----------------------------
def train_skgkt_ddp(args, fold, texts, kgs, path_data):
    rank, local_rank, world_size = ddp_setup()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.quantize and rank == 0:
        print("[WARN] --quantize enabled. For H100, quantize=False is recommended.")

    if rank == 0:
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    model, tokenizer = get_skgkt_model(
        args.base_model, test=False, local_rank=local_rank,
        pt_model_name=args.pt_model_name, r=args.r,
        lora_alpha=args.lora_alpha, quantize=args.quantize,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
    )

    train_df, val_df, _ = load_annotated_data(args, path_data)
    train_dataset = SKGKTDatasetPacked(train_df, texts, kgs, tokenizer, args)
    val_dataset = SKGKTDatasetPacked(val_df, texts, kgs, tokenizer, args)
    collator = SKGKTCollatorPacked(tokenizer, max_length=args.max_length)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )

    true_token, false_token = get_true_false_tokens(tokenizer)

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.wd, relative_step=False)

    best_key = None
    best_val_loss = None
    model_base = args.model_name + (f"_{fold}" if fold else "")
    history_path = f"results/history_{model_base}.jsonl"

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        it = tqdm(train_loader, desc=f"[R0] Training E{epoch+1}") if rank == 0 else train_loader
        for step, batch in enumerate(it):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _, _ = get_skgkt_loss_packed(model.module, batch, true_token, false_token, args)

            total_train_loss += float(loss.item())
            (loss / args.grad_accum_steps).backward()

            do_step = ((step + 1) % args.grad_accum_steps == 0) or (step == len(train_loader) - 1)
            if do_step:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss, val_auc, val_acc, val_f1 = eval_on_loader_ddp(
            model.module, val_loader, true_token, false_token, args, local_rank
        )

        if rank == 0:
            epoch_name = f"{model_base}_epoch{epoch+1:02d}"
            epoch_dir = get_checkpoint_path(epoch_name)
            model.module.save_pretrained(epoch_dir)
            print(f"[Checkpoint] Saved epoch {epoch+1} -> {epoch_dir}")

        curr_key = (val_auc, val_acc, val_f1)
        is_best = False
        if rank == 0:
            if best_key is None or curr_key > best_key:
                best_key = curr_key
                best_val_loss = val_loss
                is_best = True

        flag = torch.tensor([1 if is_best else 0], device=torch.device("cuda", local_rank), dtype=torch.int32)
        dist.broadcast(flag, src=0)

        if rank == 0:
            print(f"Epoch {epoch+1}: ValLoss={val_loss:.4f} | ValAUC={val_auc:.2f} ValACC={val_acc:.2f} ValF1={val_f1:.2f}")
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1, "val_loss": float(val_loss),
                    "val_auc": float(val_auc), "val_acc": float(val_acc), "val_f1": float(val_f1),
                    "epoch_dir": get_checkpoint_path(f"{model_base}_epoch{epoch+1:02d}")
                }, ensure_ascii=False) + "\n")

            if flag.item() == 1:
                best_dir = get_checkpoint_path(model_base)
                model.module.save_pretrained(best_dir)
                print(f"[BEST] Saved best -> {best_dir} | best_key={best_key} | best_val_loss={best_val_loss:.4f}")

    if rank == 0:
        print("\nTraining finished. Best key:", best_key)
        print("[TEST] Running rank0 test with BEST checkpoint...")
        test_skgkt(args, fold, texts, kgs, path_data, local_rank=local_rank)

    ddp_cleanup()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = build_parser()

    if len(os.sys.argv) == 1:
        argv = ["train"]
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    if args.cmd is None:
        args.cmd = "train"

    # ----- Paths -----
    path_text = "./Annonation/datasets/xes3g5ml/text_info.jsonl"
    path_data = "./Annonation/datasets/xes3g5ml/xes3g5m_large.csv"
    path_kg = "./Annonation/datasets/xes3g5ml/kg_all.json"

    with open(path_kg, "r", encoding="utf-8") as f:
        kgs = json.load(f)
    texts = read_jsonl(path_text)

    fold = "xes3g5mL_train2"

    if args.cmd == "train":
        if is_torchrun():
            train_skgkt_ddp(args, fold, texts, kgs, path_data)
        else:
            print("[INFO] Single-GPU mode (no torchrun detected).")
            train_skgkt_single(args, fold, texts, kgs, path_data)
    else:
        raise ValueError(f"Unsupported cmd={args.cmd}. Use `train`.")


if __name__ == "__main__":
    main()
