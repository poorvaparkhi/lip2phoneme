#!/usr/bin/env python3
"""
Run phoneme inference over every sample in val.csv and report validation-set
averages at the end.

This script intentionally has NO command-line arguments. Edit only the CONFIG
section below, then run:

    python infer_val_lip2phoneme.py

It imports Lip2PhonemeDataset from dataset.py so the crop loading and
normalization are exactly the same as validation during training.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_packed import TinyLip2PhonemeCTC
from dataset import Lip2PhonemeDataset, collate_lip_phone_batch


# =============================================================================
# CONFIG — edit paths/settings here. No command-line arguments are used.
# =============================================================================
VAL_CSV = "/media/newhddd/poorva/lip2phoneme/val.csv"
PHONE_VOCAB_JSON = "/media/newhddd/poorva/lip2phoneme/phone_vocab.json"
CHECKPOINT_PATH = "/media/newhddd/poorva/lip2phoneme/best_lip2phoneme_temporal_cnn_bilstm_ctc.pt"

# Per-sample results. This is overwritten each run.
OUTPUT_JSONL = "val_inference_predictions.jsonl"

# Dataset/model settings: must match the checkpoint's training configuration.
BATCH_SIZE = 8
NUM_WORKERS = 2

HIDDEN_DIM = 128
POOL_SIZE = 3
TEMPORAL_DIM = 192
TEMPORAL_DROPOUT = 0.10
RNN_DROPOUT = 0.20

BLANK_ID = 0
BEAM_WIDTH = 10
TOKEN_PRUNE = 15

# "auto" chooses CUDA when available; otherwise CPU.
DEVICE = "auto"
# =============================================================================


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"DEVICE={device_name!r}, but CUDA is not available. "
            "Set DEVICE='auto' or DEVICE='cpu'."
        )
    return device


# -----------------------------------------------------------------------------
# CTC prefix beam search — exactly the decoder used in validation during train.
# -----------------------------------------------------------------------------
def logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) for ordinary Python floats."""
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
    Decode one utterance of shape [num_real_ctc_frames, num_phone_classes].

    Output is already CTC-collapsed:
      - blanks are removed
      - repeated phones only survive when separated by a blank
    """
    if sample_log_probs.ndim != 2:
        raise ValueError(
            f"Expected [L, C] log probabilities; got {tuple(sample_log_probs.shape)}."
        )

    if sample_log_probs.size(0) == 0:
        return []
    if beam_width < 1 or token_prune < 1:
        raise ValueError("beam_width and token_prune must both be >= 1.")

    # prefix -> (log P(prefix ending in blank), log P(prefix ending in nonblank))
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        (): (0.0, -math.inf)
    }

    num_classes = sample_log_probs.size(1)
    top_k = min(token_prune, num_classes)

    for time_idx in range(sample_log_probs.size(0)):
        frame = sample_log_probs[time_idx]

        # Expand only likely phones, but always include the CTC blank.
        candidate_ids = torch.topk(frame, k=top_k).indices.tolist()
        if blank_id not in candidate_ids:
            candidate_ids.append(blank_id)

        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def add_probability(
            prefix: Tuple[int, ...],
            ends_in_blank: bool,
            score: float,
        ) -> None:
            previous_blank, previous_nonblank = next_beams.get(
                prefix,
                (-math.inf, -math.inf),
            )
            if ends_in_blank:
                next_beams[prefix] = (
                    logaddexp(previous_blank, score),
                    previous_nonblank,
                )
            else:
                next_beams[prefix] = (
                    previous_blank,
                    logaddexp(previous_nonblank, score),
                )

        blank_logp = float(frame[blank_id].item())

        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = logaddexp(p_blank, p_nonblank)

            # Emit a blank. Collapsed prefix stays unchanged.
            add_probability(
                prefix=prefix,
                ends_in_blank=True,
                score=p_total + blank_logp,
            )

            for token_id in candidate_ids:
                if token_id == blank_id:
                    continue

                token_logp = float(frame[token_id].item())
                last_token = prefix[-1] if prefix else None

                if token_id == last_token:
                    # p -> p remains the same CTC-collapsed prefix.
                    add_probability(
                        prefix=prefix,
                        ends_in_blank=False,
                        score=p_nonblank + token_logp,
                    )

                    # p -> blank -> p becomes a true repeated phone: [p, p].
                    add_probability(
                        prefix=prefix + (token_id,),
                        ends_in_blank=False,
                        score=p_blank + token_logp,
                    )
                else:
                    add_probability(
                        prefix=prefix + (token_id,),
                        ends_in_blank=False,
                        score=p_total + token_logp,
                    )

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
) -> List[List[str]]:
    """
    Decode [T, B, C] output while respecting each unpadded output length.
    """
    log_probs = log_probs.detach().cpu()
    output_lens = output_lens.detach().cpu()

    _, batch_size, _ = log_probs.shape
    predictions: List[List[str]] = []

    for batch_index in range(batch_size):
        output_len = int(output_lens[batch_index].item())
        token_ids = ctc_prefix_beam_search_single(
            sample_log_probs=log_probs[:output_len, batch_index, :],
            blank_id=BLANK_ID,
            beam_width=BEAM_WIDTH,
            token_prune=TOKEN_PRUNE,
        )
        predictions.append([id_to_phone[token_id] for token_id in token_ids])

    return predictions


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def split_targets(
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    id_to_phone: Dict[int, str],
) -> List[List[str]]:
    """Split the concatenated CTC target tensor into per-utterance references."""
    references: List[List[str]] = []
    offset = 0
    targets = targets.detach().cpu()

    for target_len in target_lens.detach().cpu():
        length = int(target_len.item())
        target_ids = targets[offset : offset + length].tolist()
        references.append([id_to_phone[token_id] for token_id in target_ids])
        offset += length

    return references


def edit_distance_ops(
    ref: Sequence[str],
    hyp: Sequence[str],
) -> Tuple[int, int, int]:
    """
    Return (substitutions, deletions, insertions) for one reference/hypothesis.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,                    # deletion
                dp[i][j - 1] + 1,                    # insertion
                dp[i - 1][j - 1] + substitution_cost,
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


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Mean/min/max for a list. Values are numeric per-sample statistics.
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def format_stat(stat: Dict[str, float], decimals: int = 2) -> str:
    return (
        f"mean={stat['mean']:.{decimals}f} | "
        f"min={stat['min']:.{decimals}f} | "
        f"max={stat['max']:.{decimals}f}"
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def load_model(
    device: torch.device,
) -> Tuple[TinyLip2PhonemeCTC, Dict[int, str], dict]:
    checkpoint_file = Path(CHECKPOINT_PATH)
    vocab_file = Path(PHONE_VOCAB_JSON)

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file.resolve()}")
    if not vocab_file.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_file.resolve()}")

    with vocab_file.open("r", encoding="utf-8") as f:
        phone_to_id = {phone: int(token_id) for phone, token_id in json.load(f).items()}

    id_to_phone = {token_id: phone for phone, token_id in phone_to_id.items()}

    if BLANK_ID not in id_to_phone:
        raise ValueError(
            f"BLANK_ID={BLANK_ID} is absent from phone_vocab.json."
        )

    # Explicit weights_only=False makes the script compatible with checkpoints
    # that contain normal Python metadata as well as tensors.
    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Fallback for older PyTorch releases that do not support weights_only.
        checkpoint = torch.load(checkpoint_file, map_location=device)

    checkpoint_vocab = checkpoint.get("phone_to_id")
    if checkpoint_vocab is not None:
        checkpoint_vocab = {
            phone: int(token_id) for phone, token_id in checkpoint_vocab.items()
        }
        if checkpoint_vocab != phone_to_id:
            raise ValueError(
                "phone_vocab.json does not match the vocabulary stored in the checkpoint. "
                "Use the vocabulary from the exact training run."
            )

    model = TinyLip2PhonemeCTC(
        num_classes=len(phone_to_id),
        hidden_dim=HIDDEN_DIM,
        pooled_size=POOL_SIZE,
        temporal_dim=TEMPORAL_DIM,
        temporal_dropout=TEMPORAL_DROPOUT,
        rnn_dropout=RNN_DROPOUT,
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint does not match the architecture configured above. "
            "Verify HIDDEN_DIM, POOL_SIZE, TEMPORAL_DIM, TEMPORAL_DROPOUT, and "
            "RNN_DROPOUT against the training run."
        ) from exc

    model.eval()
    return model, id_to_phone, checkpoint


# -----------------------------------------------------------------------------
# Main inference
# -----------------------------------------------------------------------------
@torch.inference_mode()
def main() -> None:
    device = resolve_device(DEVICE)
    output_file = Path(OUTPUT_JSONL)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    val_csv_file = Path(VAL_CSV)
    if not val_csv_file.is_file():
        raise FileNotFoundError(f"val.csv not found: {val_csv_file.resolve()}")

    model, id_to_phone, checkpoint = load_model(device)

    # This reads every sample specified in val.csv and applies the same
    # loading/normalization code used by the training validation loop.
    val_ds = Lip2PhonemeDataset(
        metadata_csv=VAL_CSV,
        phone_vocab_json=PHONE_VOCAB_JSON,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_lip_phone_batch,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation CSV: {VAL_CSV}")
    print(f"Samples in val.csv: {len(val_ds)}")
    print(f"Writing per-sample results: {output_file}")

    input_frame_values: List[float] = []
    ctc_frame_values: List[float] = []
    ref_length_values: List[float] = []
    hyp_length_values: List[float] = []
    per_values: List[float] = []
    hyp_ref_ratio_values: List[float] = []

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_ref_phones = 0
    total_hyp_phones = 0
    successful_samples = 0

    with output_file.open("w", encoding="utf-8") as output_handle:
        for batch_index, batch in enumerate(val_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            input_lens = batch["input_lens"].to(device, non_blocking=True)

            log_probs, output_lens = model(video, input_lens)

            hypotheses = ctc_beam_decode(
                log_probs=log_probs,
                output_lens=output_lens,
                id_to_phone=id_to_phone,
            )
            references = split_targets(
                targets=batch["target"],
                target_lens=batch["target_lens"],
                id_to_phone=id_to_phone,
            )

            utt_ids = batch.get("utt_ids", [""] * len(references))
            crop_paths = batch.get("crop_paths", [""] * len(references))
            batch_input_lens = batch["input_lens"].detach().cpu().tolist()
            batch_output_lens = output_lens.detach().cpu().tolist()

            for (
                utt_id,
                crop_path,
                input_frames,
                output_frames,
                reference,
                hypothesis,
            ) in zip(
                utt_ids,
                crop_paths,
                batch_input_lens,
                batch_output_lens,
                references,
                hypotheses,
            ):
                substitutions, deletions, insertions = edit_distance_ops(
                    reference,
                    hypothesis,
                )
                edits = substitutions + deletions + insertions
                ref_len = len(reference)
                hyp_len = len(hypothesis)
                per = edits / max(ref_len, 1)
                hyp_ref_ratio = hyp_len / max(ref_len, 1)

                record = {
                    "utt_id": utt_id,
                    "crop_path": crop_path,
                    "input_frames": int(input_frames),
                    "ctc_output_frames": int(output_frames),
                    "ref_len": ref_len,
                    "hyp_len": hyp_len,
                    "per": per,
                    "hyp_ref_len_ratio": hyp_ref_ratio,
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "reference": " ".join(reference),
                    "prediction": " ".join(hypothesis),
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                input_frame_values.append(float(input_frames))
                ctc_frame_values.append(float(output_frames))
                ref_length_values.append(float(ref_len))
                hyp_length_values.append(float(hyp_len))
                per_values.append(per)
                hyp_ref_ratio_values.append(hyp_ref_ratio)

                total_substitutions += substitutions
                total_deletions += deletions
                total_insertions += insertions
                total_ref_phones += ref_len
                total_hyp_phones += hyp_len
                successful_samples += 1

            if batch_index % 10 == 0 or batch_index == len(val_loader):
                print(
                    f"Processed batch {batch_index}/{len(val_loader)} "
                    f"({successful_samples}/{len(val_ds)} samples)"
                )

    # Corpus PER pools all phones before dividing, unlike mean(per_values), which
    # gives every utterance equal weight regardless of reference length.
    total_edits = total_substitutions + total_deletions + total_insertions
    corpus_per = total_edits / max(total_ref_phones, 1)
    corpus_sub_rate = total_substitutions / max(total_ref_phones, 1)
    corpus_del_rate = total_deletions / max(total_ref_phones, 1)
    corpus_ins_rate = total_insertions / max(total_ref_phones, 1)
    corpus_hyp_ref_ratio = total_hyp_phones / max(total_ref_phones, 1)

    # Also save the final aggregate summary beside the JSONL predictions.
    summary = {
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_per": checkpoint.get("val_per"),
        "validation_csv": VAL_CSV,
        "samples": successful_samples,
        "decoder": {
            "type": "prefix_beam_no_lm",
            "beam_width": BEAM_WIDTH,
            "token_prune": TOKEN_PRUNE,
            "blank_id": BLANK_ID,
        },
        "per_sample_averages": {
            "input_frames": summarize(input_frame_values),
            "ctc_output_frames": summarize(ctc_frame_values),
            "reference_phonemes": summarize(ref_length_values),
            "predicted_phonemes": summarize(hyp_length_values),
            "per": summarize(per_values),
            "hyp_ref_len_ratio": summarize(hyp_ref_ratio_values),
        },
        "corpus_metrics": {
            "reference_phonemes": total_ref_phones,
            "predicted_phonemes": total_hyp_phones,
            "substitutions": total_substitutions,
            "deletions": total_deletions,
            "insertions": total_insertions,
            "per": corpus_per,
            "sub_rate": corpus_sub_rate,
            "del_rate": corpus_del_rate,
            "ins_rate": corpus_ins_rate,
            "hyp_ref_len_ratio": corpus_hyp_ref_ratio,
        },
    }

    summary_file = output_file.with_suffix(".summary.json")
    with summary_file.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("VALIDATION INFERENCE SUMMARY")
    print("=" * 72)
    print(f"Samples processed:                    {successful_samples}")
    print(f"Checkpoint epoch:                     {checkpoint.get('epoch', 'unknown')}")
    print(f"Per-sample input frames:              {format_stat(summarize(input_frame_values))}")
    print(f"Per-sample CTC output frames:         {format_stat(summarize(ctc_frame_values))}")
    print(f"Per-sample reference phonemes:        {format_stat(summarize(ref_length_values))}")
    print(f"Per-sample predicted phonemes:        {format_stat(summarize(hyp_length_values))}")
    print(f"Per-sample PER:                       {format_stat(summarize(per_values), decimals=4)}")
    print(f"Per-sample hyp/ref length ratio:      {format_stat(summarize(hyp_ref_ratio_values), decimals=4)}")
    print("-" * 72)
    print(f"Corpus reference phones:              {total_ref_phones}")
    print(f"Corpus predicted phones:              {total_hyp_phones}")
    print(f"Corpus PER:                           {corpus_per:.4f}")
    print(f"Corpus substitution rate:             {corpus_sub_rate:.4f}")
    print(f"Corpus deletion rate:                 {corpus_del_rate:.4f}")
    print(f"Corpus insertion rate:                {corpus_ins_rate:.4f}")
    print(f"Corpus hyp/ref length ratio:          {corpus_hyp_ref_ratio:.4f}")
    print("-" * 72)
    print(f"Per-sample predictions:               {output_file}")
    print(f"Aggregate summary:                    {summary_file}")


if __name__ == "__main__":
    main()
