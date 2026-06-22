import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import json
import math


from model_packed import TinyLip2PhonemeCTC
from dataset import Lip2PhonemeDataset, collate_lip_phone_batch


from pathlib import Path

debug_path = Path("bad_batches_debug.jsonl")

# ----------------------------
# Config
# ----------------------------

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
PHONE_VOCAB_JSON = "phone_vocab.json"

with open("phone_vocab.json", "r", encoding="utf-8") as f:
    phone_to_id = json.load(f)

id_to_phone = {v: k for k, v in phone_to_id.items()}

BATCH_SIZE = 8
NUM_WORKERS = 2
HIDDEN_DIM = 128
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 700
GRAD_CLIP = 5.0
BLANK_ID = 0

device = "cuda" if torch.cuda.is_available() else "cpu"



def log_bad_batch(
    debug_path,
    epoch,
    step,
    reason,
    batch,
    loss=None,
    grad_norm=None,
):
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
# Metrics / decoding
# ----------------------------

def ctc_greedy_decode(log_probs, output_lens, id_to_phone, blank_id=0):
    """
    Greedy CTC decode.

    log_probs: [T, B, C]
    output_lens: [B]

    Important: decode only up to output_lens[b], not the padded batch length T.
    """
    pred_ids = torch.argmax(log_probs, dim=-1)  # [T, B]
    pred_ids = pred_ids.cpu()
    output_lens = output_lens.cpu()

    _, B = pred_ids.shape
    results = []

    for b in range(B):
        seq = []
        prev = None

        L = int(output_lens[b].item())

        for t in range(L):
            idx = int(pred_ids[t, b])

            # CTC collapse rule:
            # 1. collapse repeated symbols
            # 2. remove blanks
            if idx != blank_id and idx != prev:
                seq.append(id_to_phone[idx])

            prev = idx

        results.append(seq)

    return results


def split_targets(targets, target_lens, id_to_phone):
    refs = []
    offset = 0
    targets = targets.cpu()

    for L in target_lens:
        L = int(L.item())
        ids = targets[offset:offset + L].tolist()
        refs.append([id_to_phone[i] for i in ids])
        offset += L

    return refs


def edit_distance(ref, hyp):
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
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution or match
            )

    return dp[n][m]


def single_per(ref, hyp):
    return edit_distance(ref, hyp) / max(len(ref), 1)


def edit_distance_ops(ref, hyp):
    """
    Return substitution, deletion, and insertion counts for one ref/hyp pair.
    These sum to the edit distance.
    """
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
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution or match
            )

    i, j = n, m
    subs = 0
    dels = 0
    ins = 0

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref[i - 1] == hyp[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1

        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            subs += 1
            i -= 1
            j -= 1

        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1

        else:
            ins += 1
            j -= 1

    return subs, dels, ins


def phoneme_error_rate(refs, hyps):
    total_edits = 0
    total_len = 0

    for ref, hyp in zip(refs, hyps):
        total_edits += edit_distance(ref, hyp)
        total_len += len(ref)

    return total_edits / max(total_len, 1)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------
# Train / validate functions
# ----------------------------

grad_spike_threshold = 50.0
high_loss_threshold = 10.0


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(train_loader):
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        # ----------------------------
        # Basic CTC sanity checks
        # ----------------------------

        # If target is longer than available input frames, CTC cannot align properly.
        if (input_lens < target_lens).any():
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="input_len_less_than_target_len",
                batch=batch,
            )
            continue

        optimizer.zero_grad(set_to_none=True)


        log_probs, output_lens = model(video, input_lens)

        # More correct check: CTC uses output_lens, not raw input_lens.
        if (output_lens < target_lens).any():
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="output_len_less_than_target_len",
                batch=batch,
            )
            continue

        loss = criterion(
            log_probs,
            targets,
            output_lens,
            target_lens,
        )

        # ----------------------------
        # Log NaN / Inf loss
        # ----------------------------

        if not torch.isfinite(loss):
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="nan_or_inf_loss",
                batch=batch,
                loss=loss.detach().cpu().item(),
            )
            continue

        loss_value = loss.detach().cpu().item()

        # ----------------------------
        # Log very high loss batches
        # ----------------------------

        if loss_value > high_loss_threshold:
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="high_loss",
                batch=batch,
                loss=loss_value,
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if torch.is_tensor(grad_norm):
            grad_norm_value = grad_norm.detach().cpu().item()
        else:
            grad_norm_value = float(grad_norm)

        # ----------------------------
        # Log NaN / Inf gradient norm
        # ----------------------------

        if not math.isfinite(grad_norm_value):
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="nan_or_inf_grad_norm",
                batch=batch,
                loss=loss_value,
                grad_norm=grad_norm_value,
            )
            continue

        # ----------------------------
        # Log gradient spike batches
        # ----------------------------

        if grad_norm_value > grad_spike_threshold:
            log_bad_batch(
                debug_path=debug_path,
                epoch=epoch,
                step=step,
                reason="high_grad_norm",
                batch=batch,
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
                f"Epoch {epoch} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss {loss_value:.4f} | "
                f"GradNorm {grad_norm_value:.4f}"
            )

    return total_loss / max(total_batches, 1)



@torch.no_grad()
def validate(model, val_loader, criterion, id_to_phone):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_refs = []
    all_hyps = []

    example_rows = []
    worst_rows = []

    for batch in val_loader:
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        log_probs, output_lens = model(video, input_lens)

        loss = criterion(
            log_probs,
            targets,
            output_lens,
            target_lens,
        )

        total_loss += loss.item()
        total_batches += 1

        # IMPORTANT: decode only up to output_lens, not full padded T
        hyps = ctc_greedy_decode(
            log_probs=log_probs,
            output_lens=output_lens,
            id_to_phone=id_to_phone,
            blank_id=BLANK_ID,
        )

        refs = split_targets(
            targets=batch["target"],
            target_lens=batch["target_lens"],
            id_to_phone=id_to_phone,
        )

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

    # ----------------------------
    # Substitution / deletion / insertion breakdown
    # ----------------------------

    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_ref_len = 0

    for ref, hyp in zip(all_refs, all_hyps):
        s, d, i = edit_distance_ops(ref, hyp)

        total_subs += s
        total_dels += d
        total_ins += i
        total_ref_len += len(ref)

    sub_rate = total_subs / max(total_ref_len, 1)
    del_rate = total_dels / max(total_ref_len, 1)
    ins_rate = total_ins / max(total_ref_len, 1)

    # ----------------------------
    # Normal examples table
    # ----------------------------

    examples_table = wandb.Table(
        columns=[
            "utt_id",
            "ref_len",
            "hyp_len",
            "per",
            "reference",
            "prediction",
        ],
        data=example_rows,
    )

    # ----------------------------
    # Worst validation examples table
    # ----------------------------

    worst_rows = sorted(worst_rows, key=lambda x: x["per"], reverse=True)

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
        examples_table,
        worst_examples_table,
    )

# ----------------------------
# Main training script
# ----------------------------

train_ds = Lip2PhonemeDataset(
    metadata_csv=TRAIN_CSV,
    phone_vocab_json=PHONE_VOCAB_JSON,
)

val_ds = Lip2PhonemeDataset(
    metadata_csv=VAL_CSV,
    phone_vocab_json=PHONE_VOCAB_JSON,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_lip_phone_batch,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_lip_phone_batch,
    pin_memory=True,
)

model = TinyLip2PhonemeCTC(
    num_classes=len(phone_to_id),
    hidden_dim=HIDDEN_DIM,
).to(device)

total_params, trainable_params = count_parameters(model)

print(f"Device: {device}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CTCLoss(
    blank=BLANK_ID,
    reduction="mean",
    zero_infinity=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

wandb.init(
    project="lip-to-phoneme-ctc",
    name="packed-tiny-cnn-bilstm-ctc",
    config={
        "model": model.__class__.__name__,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "blank_id": BLANK_ID,
        "num_classes": len(phone_to_id),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    },
)

wandb.watch(model, log="gradients", log_freq=100)

best_val_per = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=epoch,
    )

    (
        val_loss,
        val_per,
        sub_rate,
        del_rate,
        ins_rate,
        examples_table,
        worst_examples_table,
    ) = validate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        id_to_phone=id_to_phone,
    )

    if val_per < best_val_per:
        best_val_per = val_per

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phone_to_id": phone_to_id,
                "id_to_phone": id_to_phone,
                "val_loss": val_loss,
                "val_per": val_per,
                "val_sub_rate": sub_rate,
                "val_del_rate": del_rate,
                "val_ins_rate": ins_rate,
                "total_params": total_params,
            },
            "best_lip2phoneme_bilstmpacked_ctc_700epochs.pt",
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
            "val/examples": examples_table,
            "val/worst_examples": worst_examples_table,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PER: {val_per:.4f} | "
        f"Sub: {sub_rate:.4f} | "
        f"Del: {del_rate:.4f} | "
        f"Ins: {ins_rate:.4f} | "
        f"Best PER: {best_val_per:.4f}"
    )

wandb.finish()
