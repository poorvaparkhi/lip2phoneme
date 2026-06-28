import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv
import random

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Sampler

from model_packed import TinyLip2PhonemeCTC
from dataset import Lip2PhonemeDataset, collate_lip_phone_batch


# ----------------------------
# Config
# ----------------------------
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
PHONE_VOCAB_JSON = "phone_vocab.json"

BATCH_SIZE = 8
NUM_WORKERS = 2
HIDDEN_DIM = 128
POOL_SIZE = 3                 # preserves 3x3 coarse spatial layout
TEMPORAL_DIM = 192
TEMPORAL_DROPOUT = 0.10
RNN_DROPOUT = 0.20
TSM_FOLD_DIV = 8
BUCKET_SIZE = 64              # bigger bucket = more shuffle, smaller bucket = less padding

LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 500
GRAD_CLIP = 5.0
BLANK_ID = 0

# Prefix beam search. This is a CTC beam search without a phoneme LM.
BEAM_WIDTH = 10
TOKEN_PRUNE = 15              # keep only the best 15 phones per CTC timestep

# Do not let ReduceLROnPlateau reduce LR during the early CTC blank-heavy phase.
SCHEDULER_START_EPOCH = 180
SCHEDULER_PATIENCE = 30
SCHEDULER_FACTOR = 0.5
SCHEDULER_THRESHOLD = 0.001   # absolute PER improvement required to reset patience
MIN_LR = 1e-6

CHECKPOINT_PATH = "best_lip2phoneme_tsm_bilstm_ctc.pt"
debug_path = Path("bad_batches_debug.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(PHONE_VOCAB_JSON, "r", encoding="utf-8") as f:
    phone_to_id = json.load(f)
id_to_phone = {v: k for k, v in phone_to_id.items()}




# ----------------------------
# Length-bucketed batching
# ----------------------------
class LengthBucketBatchSampler(Sampler[List[int]]):
    """
    Groups examples with similar frame lengths into the same batch.

    This is especially useful for the TSM model because padded frames are present
    in the [B, T, C, H, W] tensor that enters CNN BatchNorm. Bucketing reduces
    the number of padded zero frames inside each batch.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if bucket_size < batch_size:
            raise ValueError("bucket_size should be >= batch_size")

        self.lengths = [int(length) for length in lengths]
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            random.shuffle(indices)

        buckets = [
            indices[i : i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        batches = []
        for bucket in buckets:
            bucket.sort(key=lambda idx: self.lengths[idx])

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def infer_lengths_from_csv(metadata_csv: str) -> Optional[List[int]]:
    """
    Prefer cheap length extraction from metadata CSV.

    Supported cases:
      1. input_len / num_frames / n_frames / frames column exists
      2. duration and fps columns exist

    Returns None if the CSV does not contain enough information.
    """
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    frame_columns = ["input_len", "num_frames", "n_frames", "frames", "T"]
    for column in frame_columns:
        if column in rows[0] and rows[0][column] not in (None, ""):
            return [int(round(float(row[column]))) for row in rows]

    if "duration" in rows[0] and "fps" in rows[0]:
        return [
            max(1, int(round(float(row["duration"]) * float(row["fps"]))))
            for row in rows
        ]

    return None


def get_dataset_lengths(dataset: Lip2PhonemeDataset, metadata_csv: str) -> List[int]:
    """
    Get frame lengths for bucketed batching.

    First tries metadata CSV to avoid loading every .npy. If unavailable, falls
    back to dataset[i]["video"].shape[0].
    """
    lengths = infer_lengths_from_csv(metadata_csv)
    if lengths is not None:
        if len(lengths) != len(dataset):
            raise ValueError(
                f"Length count mismatch: CSV has {len(lengths)}, dataset has {len(dataset)}"
            )
        return lengths

    loaded_lengths = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if "input_len" in sample:
            loaded_lengths.append(int(sample["input_len"]))
        elif "input_lens" in sample:
            loaded_lengths.append(int(sample["input_lens"]))
        else:
            loaded_lengths.append(int(sample["video"].shape[0]))
    return loaded_lengths


# ----------------------------
# Debug logging
# ----------------------------
def log_bad_batch(
    debug_path: Path,
    epoch: int,
    step: int,
    reason: str,
    batch: Dict,
    loss: Optional[float] = None,
    grad_norm: Optional[float] = None,
) -> None:
    record = {
        "epoch": int(epoch),
        "step": int(step),
        "reason": reason,
        "utt_ids": batch["utt_ids"],
        "input_lens": batch["input_lens"].detach().cpu().tolist(),
        "target_lens": batch["target_lens"].detach().cpu().tolist(),
    }
    if "crop_paths" in batch:
        record["crop_paths"] = batch["crop_paths"]
    if loss is not None:
        record["loss"] = float(loss)
    if grad_norm is not None:
        record["grad_norm"] = float(grad_norm)

    with open(debug_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# CTC prefix beam search
# ----------------------------
def logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) for Python floats."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def ctc_prefix_beam_search_single(
    sample_log_probs: torch.Tensor,
    blank_id: int,
    beam_width: int,
    token_prune: int,
) -> List[int]:
    """
    Prefix beam search for one utterance.

    Args:
        sample_log_probs: [L, C] log probabilities, only real CTC frames.
        blank_id:         CTC blank token ID.
        beam_width:       number of prefixes retained after every frame.
        token_prune:      number of high-probability symbols expanded per frame.

    Returns:
        Best collapsed CTC token-ID sequence. No language model is used.
    """
    if sample_log_probs.ndim != 2:
        raise ValueError(
            f"Expected [L, C] log probabilities, got {tuple(sample_log_probs.shape)}"
        )
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if token_prune < 1:
        raise ValueError("token_prune must be >= 1")

    # prefix -> (log P(prefix ends in blank), log P(prefix ends in non-blank))
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        (): (0.0, -math.inf)
    }

    num_classes = sample_log_probs.size(1)
    top_k = min(token_prune, num_classes)

    for t in range(sample_log_probs.size(0)):
        frame = sample_log_probs[t]
        top_ids = torch.topk(frame, k=top_k).indices.tolist()
        if blank_id not in top_ids:
            top_ids.append(blank_id)

        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def add_probability(prefix: Tuple[int, ...], is_blank: bool, value: float) -> None:
            old_blank, old_nonblank = next_beams.get(
                prefix, (-math.inf, -math.inf)
            )
            if is_blank:
                next_beams[prefix] = (logaddexp(old_blank, value), old_nonblank)
            else:
                next_beams[prefix] = (old_blank, logaddexp(old_nonblank, value))

        blank_logp = float(frame[blank_id].item())

        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = logaddexp(p_blank, p_nonblank)

            # Emit blank: prefix stays unchanged.
            add_probability(prefix, is_blank=True, value=p_total + blank_logp)

            for token_id in top_ids:
                if token_id == blank_id:
                    continue

                token_logp = float(frame[token_id].item())
                last_token = prefix[-1] if prefix else None

                if token_id == last_token:
                    # Repeating a token without an intervening blank keeps the
                    # same collapsed prefix, and can only originate from p_nonblank.
                    add_probability(
                        prefix,
                        is_blank=False,
                        value=p_nonblank + token_logp,
                    )

                    # A blank before the same phone creates a true repeated phone:
                    # e.g. p, blank, p -> [p, p].
                    repeated_prefix = prefix + (token_id,)
                    add_probability(
                        repeated_prefix,
                        is_blank=False,
                        value=p_blank + token_logp,
                    )
                else:
                    extended_prefix = prefix + (token_id,)
                    add_probability(
                        extended_prefix,
                        is_blank=False,
                        value=p_total + token_logp,
                    )

        # Keep only the highest-probability prefixes before advancing in time.
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: logaddexp(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix, _ = max(
        beams.items(),
        key=lambda item: logaddexp(item[1][0], item[1][1]),
    )
    return list(best_prefix)


def ctc_beam_decode(
    log_probs: torch.Tensor,
    output_lens: torch.Tensor,
    id_to_phone: Dict[int, str],
    blank_id: int = 0,
    beam_width: int = 10,
    token_prune: int = 15,
) -> List[List[str]]:
    """Beam-decode [T,B,C] CTC output, respecting every utterance length."""
    log_probs = log_probs.detach().cpu()
    output_lens = output_lens.detach().cpu()

    _, batch_size, _ = log_probs.shape
    results: List[List[str]] = []

    for b in range(batch_size):
        length = int(output_lens[b].item())
        token_ids = ctc_prefix_beam_search_single(
            sample_log_probs=log_probs[:length, b, :],
            blank_id=blank_id,
            beam_width=beam_width,
            token_prune=token_prune,
        )
        results.append([id_to_phone[token_id] for token_id in token_ids])

    return results


# ----------------------------
# Metrics
# ----------------------------
def split_targets(
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    id_to_phone: Dict[int, str],
) -> List[List[str]]:
    refs = []
    offset = 0
    targets = targets.detach().cpu()

    for length in target_lens:
        length = int(length.item())
        ids = targets[offset : offset + length].tolist()
        refs.append([id_to_phone[token_id] for token_id in ids])
        offset += length

    return refs


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def single_per(ref: Sequence[str], hyp: Sequence[str]) -> float:
    return edit_distance(ref, hyp) / max(len(ref), 1)


def edit_distance_ops(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Return substitutions, deletions, and insertions for one pair."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = n, m
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref[i - 1] == hyp[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return substitutions, deletions, insertions


def phoneme_error_rate(refs: Sequence[Sequence[str]], hyps: Sequence[Sequence[str]]) -> float:
    total_edits = sum(edit_distance(ref, hyp) for ref, hyp in zip(refs, hyps))
    total_ref_len = sum(len(ref) for ref in refs)
    return total_edits / max(total_ref_len, 1)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


# ----------------------------
# Train / validation
# ----------------------------
grad_spike_threshold = 50.0
high_loss_threshold = 10.0


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(train_loader):
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        if (input_lens < target_lens).any():
            log_bad_batch(
                debug_path, epoch, step, "input_len_less_than_target_len", batch
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        log_probs, output_lens = model(video, input_lens)

        if (output_lens < target_lens).any():
            log_bad_batch(
                debug_path, epoch, step, "output_len_less_than_target_len", batch
            )
            continue

        loss = criterion(log_probs, targets, output_lens, target_lens)
        if not torch.isfinite(loss):
            log_bad_batch(
                debug_path,
                epoch,
                step,
                "nan_or_inf_loss",
                batch,
                loss=float(loss.detach().cpu().item()),
            )
            continue

        loss_value = float(loss.detach().cpu().item())
        if loss_value > high_loss_threshold:
            log_bad_batch(debug_path, epoch, step, "high_loss", batch, loss=loss_value)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norm_value = float(grad_norm.detach().cpu().item())

        if not math.isfinite(grad_norm_value):
            log_bad_batch(
                debug_path,
                epoch,
                step,
                "nan_or_inf_grad_norm",
                batch,
                loss=loss_value,
                grad_norm=grad_norm_value,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if grad_norm_value > grad_spike_threshold:
            log_bad_batch(
                debug_path,
                epoch,
                step,
                "high_grad_norm",
                batch,
                loss=loss_value,
                grad_norm=grad_norm_value,
            )

        optimizer.step()
        total_loss += loss_value
        total_batches += 1

        if step % 20 == 0:
            wandb.log(
                {
                    "train/step_loss": loss_value,
                    "train/grad_norm": grad_norm_value,
                    "epoch": epoch,
                }
            )
            print(
                f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                f"Loss {loss_value:.4f} | GradNorm {grad_norm_value:.4f}"
            )

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CTCLoss,
    id_to_phone: Dict[int, str],
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_refs: List[List[str]] = []
    all_hyps: List[List[str]] = []
    example_rows = []
    worst_rows = []

    for batch in val_loader:
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        log_probs, output_lens = model(video, input_lens)
        loss = criterion(log_probs, targets, output_lens, target_lens)
        total_loss += float(loss.item())
        total_batches += 1

        # Beam search replaces greedy CTC decoding for all validation metrics.
        hyps = ctc_beam_decode(
            log_probs=log_probs,
            output_lens=output_lens,
            id_to_phone=id_to_phone,
            blank_id=BLANK_ID,
            beam_width=BEAM_WIDTH,
            token_prune=TOKEN_PRUNE,
        )
        refs = split_targets(batch["target"], batch["target_lens"], id_to_phone)

        all_refs.extend(refs)
        all_hyps.extend(hyps)

        utt_ids = batch.get("utt_ids", [""] * len(refs))
        crop_paths = batch.get("crop_paths", [""] * len(refs))

        for utt_id, crop_path, ref, hyp in zip(utt_ids, crop_paths, refs, hyps):
            per = single_per(ref, hyp)
            worst_rows.append(
                {
                    "utt_id": utt_id,
                    "crop_path": crop_path,
                    "per": per,
                    "ref_len": len(ref),
                    "hyp_len": len(hyp),
                    "reference": " ".join(ref),
                    "prediction": " ".join(hyp),
                }
            )
            if len(example_rows) < 10:
                example_rows.append(
                    [
                        utt_id,
                        len(ref),
                        len(hyp),
                        per,
                        " ".join(ref),
                        " ".join(hyp),
                    ]
                )

    val_loss = total_loss / max(total_batches, 1)
    val_per = phoneme_error_rate(all_refs, all_hyps)

    total_subs = total_dels = total_ins = total_ref_len = total_hyp_len = 0
    for ref, hyp in zip(all_refs, all_hyps):
        substitutions, deletions, insertions = edit_distance_ops(ref, hyp)
        total_subs += substitutions
        total_dels += deletions
        total_ins += insertions
        total_ref_len += len(ref)
        total_hyp_len += len(hyp)

    sub_rate = total_subs / max(total_ref_len, 1)
    del_rate = total_dels / max(total_ref_len, 1)
    ins_rate = total_ins / max(total_ref_len, 1)
    hyp_ref_len_ratio = total_hyp_len / max(total_ref_len, 1)

    examples_table = wandb.Table(
        columns=["utt_id", "ref_len", "hyp_len", "per", "reference", "prediction"],
        data=example_rows,
    )

    worst_rows.sort(key=lambda row: row["per"], reverse=True)
    worst_examples_table = wandb.Table(
        columns=[
            "utt_id",
            "crop_path",
            "per",
            "ref_len",
            "hyp_len",
            "reference",
            "prediction",
        ],
        data=[
            [
                row["utt_id"],
                row["crop_path"],
                row["per"],
                row["ref_len"],
                row["hyp_len"],
                row["reference"],
                row["prediction"],
            ]
            for row in worst_rows[:20]
        ],
    )

    return (
        val_loss,
        val_per,
        sub_rate,
        del_rate,
        ins_rate,
        hyp_ref_len_ratio,
        examples_table,
        worst_examples_table,
    )


# ----------------------------
# Main
# ----------------------------
train_ds = Lip2PhonemeDataset(
    metadata_csv=TRAIN_CSV,
    phone_vocab_json=PHONE_VOCAB_JSON,
    augment=True
)
val_ds = Lip2PhonemeDataset(
    metadata_csv=VAL_CSV,
    phone_vocab_json=PHONE_VOCAB_JSON,
    augment=False
)

train_lengths = get_dataset_lengths(train_ds, TRAIN_CSV)
val_lengths = get_dataset_lengths(val_ds, VAL_CSV)

train_batch_sampler = LengthBucketBatchSampler(
    lengths=train_lengths,
    batch_size=BATCH_SIZE,
    bucket_size=BUCKET_SIZE,
    shuffle=True,
    drop_last=False,
)
val_batch_sampler = LengthBucketBatchSampler(
    lengths=val_lengths,
    batch_size=BATCH_SIZE,
    bucket_size=BUCKET_SIZE,
    shuffle=False,
    drop_last=False,
)

train_loader = DataLoader(
    train_ds,
    batch_sampler=train_batch_sampler,
    num_workers=NUM_WORKERS,
    collate_fn=collate_lip_phone_batch,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_sampler=val_batch_sampler,
    num_workers=NUM_WORKERS,
    collate_fn=collate_lip_phone_batch,
    pin_memory=True,
)

model = TinyLip2PhonemeCTC(
    num_classes=len(phone_to_id),
    hidden_dim=HIDDEN_DIM,
    pooled_size=POOL_SIZE,
    temporal_dim=TEMPORAL_DIM,
    temporal_dropout=TEMPORAL_DROPOUT,
    rnn_dropout=RNN_DROPOUT,
    tsm_fold_div=TSM_FOLD_DIV,
).to(device)

total_params, trainable_params = count_parameters(model)
print(f"Device: {device}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    threshold=SCHEDULER_THRESHOLD,
    threshold_mode="abs",
    cooldown=5,
    min_lr=MIN_LR,
)

wandb.init(
    project="lip-to-phoneme-ctc",
    name="tsm-cnn-bilstm-ctc-beam",
    config={
        "model": model.__class__.__name__,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "pool_size": POOL_SIZE,
        "temporal_dim": TEMPORAL_DIM,
        "temporal_dropout": TEMPORAL_DROPOUT,
        "rnn_dropout": RNN_DROPOUT,
        "tsm_fold_div": TSM_FOLD_DIV,
        "bucket_size": BUCKET_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "blank_id": BLANK_ID,
        "num_classes": len(phone_to_id),
        "beam_width": BEAM_WIDTH,
        "token_prune": TOKEN_PRUNE,
        "scheduler_start_epoch": SCHEDULER_START_EPOCH,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_threshold": SCHEDULER_THRESHOLD,
        "min_lr": MIN_LR,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    },
)
wandb.watch(model, log="gradients", log_freq=100)

best_val_per = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

    (
        val_loss,
        val_per,
        sub_rate,
        del_rate,
        ins_rate,
        hyp_ref_len_ratio,
        examples_table,
        worst_examples_table,
    ) = validate(model, val_loader, criterion, id_to_phone)

    # Begin only after the initial CTC blank-dominant period; after that,
    # reduce LR only when beam-search PER has stopped improving.
    if epoch >= SCHEDULER_START_EPOCH:
        scheduler.step(val_per)

    current_lr = optimizer.param_groups[0]["lr"]

    if val_per < best_val_per:
        best_val_per = val_per
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "phone_to_id": phone_to_id,
                "id_to_phone": id_to_phone,
                "val_loss": val_loss,
                "val_per": val_per,
                "val_sub_rate": sub_rate,
                "val_del_rate": del_rate,
                "val_ins_rate": ins_rate,
                "val_hyp_ref_len_ratio": hyp_ref_len_ratio,
                "beam_width": BEAM_WIDTH,
                "token_prune": TOKEN_PRUNE,
                "total_params": total_params,
            },
            CHECKPOINT_PATH,
        )

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/per": val_per,
            "val/best_per": best_val_per,
            "val/sub_rate": sub_rate,
            "val/del_rate": del_rate,
            "val/ins_rate": ins_rate,
            "val/hyp_ref_len_ratio": hyp_ref_len_ratio,
            "val/decoder": "prefix_beam_no_lm",
            "lr": current_lr,
            "val/examples": examples_table,
            "val/worst_examples": worst_examples_table,
        }
    )

    print(
        f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | Beam PER: {val_per:.4f} | "
        f"Sub: {sub_rate:.4f} | Del: {del_rate:.4f} | Ins: {ins_rate:.4f} | "
        f"Hyp/Ref: {hyp_ref_len_ratio:.4f} | LR: {current_lr:.2e} | "
        f"Best PER: {best_val_per:.4f}"
    )

wandb.finish()
