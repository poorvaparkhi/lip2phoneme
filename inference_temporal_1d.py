#!/usr/bin/env python3
"""
Inference / evaluation for TinyLip2PhonemeCTC trained by the temporal CNN +
BiLSTM CTC training script.

Modes
-----
1) Infer one raw lip-crop file (.npy or .npz):
   python inference_lip2phoneme.py --checkpoint best_lip2phoneme_temporal_cnn_bilstm_ctc.pt \
       --crop /path/to/clip.npy --output one_prediction.jsonl

2) Re-run inference on every sample in a metadata CSV, using dataset.py. This
   is the recommended evaluation path because it applies the *same* loading and
   normalization used during validation/training:
   python inference_lip2phoneme.py --checkpoint best_lip2phoneme_temporal_cnn_bilstm_ctc.pt \
       --metadata-csv val.csv --phone-vocab phone_vocab.json \
       --output val_predictions.jsonl

The decoder is the same prefix beam search used by the supplied training script:
beam width 10, top-15 token pruning, and CTC blank ID 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_temporal_1d import TinyLip2PhonemeCTC


# -----------------------------------------------------------------------------
# CTC prefix beam search -- identical decoding logic to training validation.
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
    Decode one utterance of shape [L, C] into CTC-collapsed phone IDs.

    The returned sequence has no blank IDs. Consecutive repeats remain only
    when the model emitted a blank between them.
    """
    if sample_log_probs.ndim != 2:
        raise ValueError(
            f"Expected CTC log probabilities [L, C], got {tuple(sample_log_probs.shape)}."
        )
    if sample_log_probs.size(0) == 0:
        return []
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")
    if token_prune < 1:
        raise ValueError("token_prune must be >= 1.")
    if not (0 <= blank_id < sample_log_probs.size(1)):
        raise ValueError(
            f"blank_id={blank_id} is invalid for {sample_log_probs.size(1)} classes."
        )

    # prefix -> (log P(prefix ending in blank), log P(prefix ending in nonblank))
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}
    num_classes = sample_log_probs.size(1)
    top_k = min(token_prune, num_classes)

    for time_index in range(sample_log_probs.size(0)):
        frame = sample_log_probs[time_index]
        candidate_ids = torch.topk(frame, k=top_k).indices.tolist()
        # Blank must always be considered, even if it did not make the top-k.
        if blank_id not in candidate_ids:
            candidate_ids.append(blank_id)

        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def add_probability(
            prefix: Tuple[int, ...],
            ends_in_blank: bool,
            score: float,
        ) -> None:
            prev_blank, prev_nonblank = next_beams.get(prefix, (-math.inf, -math.inf))
            if ends_in_blank:
                next_beams[prefix] = (logaddexp(prev_blank, score), prev_nonblank)
            else:
                next_beams[prefix] = (prev_blank, logaddexp(prev_nonblank, score))

        blank_logp = float(frame[blank_id].item())

        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = logaddexp(p_blank, p_nonblank)

            # Emit blank: collapsed prefix is unchanged.
            add_probability(prefix, ends_in_blank=True, score=p_total + blank_logp)

            for token_id in candidate_ids:
                if token_id == blank_id:
                    continue

                token_logp = float(frame[token_id].item())
                last_token = prefix[-1] if prefix else None

                if token_id == last_token:
                    # Same phone with no blank in between stays the same prefix.
                    add_probability(
                        prefix,
                        ends_in_blank=False,
                        score=p_nonblank + token_logp,
                    )
                    # Same phone after a blank becomes a true repeated phone.
                    add_probability(
                        prefix + (token_id,),
                        ends_in_blank=False,
                        score=p_blank + token_logp,
                    )
                else:
                    add_probability(
                        prefix + (token_id,),
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
    blank_id: int,
    beam_width: int,
    token_prune: int,
) -> List[List[str]]:
    """Decode a [T, B, C] batch while ignoring padded CTC output frames."""
    log_probs = log_probs.detach().cpu()
    output_lens = output_lens.detach().cpu()

    _, batch_size, _ = log_probs.shape
    predictions: List[List[str]] = []

    for batch_index in range(batch_size):
        length = int(output_lens[batch_index].item())
        token_ids = ctc_prefix_beam_search_single(
            sample_log_probs=log_probs[:length, batch_index, :],
            blank_id=blank_id,
            beam_width=beam_width,
            token_prune=token_prune,
        )
        predictions.append([id_to_phone[token_id] for token_id in token_ids])

    return predictions


# -----------------------------------------------------------------------------
# Checkpoint/model loading
# -----------------------------------------------------------------------------
def torch_load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Explicit weights_only=False supports normal Python metadata in checkpoints
    # on PyTorch >= 2.6; the fallback supports older releases.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a checkpoint dictionary containing 'model_state_dict'."
        )
    return checkpoint


def get_phone_vocab(
    checkpoint: Dict[str, Any],
    phone_vocab_path: Path | None,
) -> Dict[int, str]:
    """Prefer the vocabulary embedded in the checkpoint."""
    phone_to_id = checkpoint.get("phone_to_id")

    if phone_to_id is None:
        if phone_vocab_path is None:
            raise ValueError(
                "Checkpoint has no 'phone_to_id'. Provide --phone-vocab."
            )
        if not phone_vocab_path.is_file():
            raise FileNotFoundError(f"Phone vocabulary not found: {phone_vocab_path}")
        with phone_vocab_path.open("r", encoding="utf-8") as handle:
            phone_to_id = json.load(handle)

    phone_to_id = {str(phone): int(index) for phone, index in phone_to_id.items()}
    id_to_phone = {index: phone for phone, index in phone_to_id.items()}

    if not id_to_phone:
        raise ValueError("Phone vocabulary is empty.")
    return id_to_phone


def build_model(
    checkpoint: Dict[str, Any],
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
) -> TinyLip2PhonemeCTC:
    # Defaults below exactly match the supplied training file.
    model = TinyLip2PhonemeCTC(
        num_classes=len(id_to_phone),
        hidden_dim=args.hidden_dim,
        pooled_size=args.pool_size,
        temporal_dim=args.temporal_dim,
        temporal_dropout=args.temporal_dropout,
        rnn_dropout=args.rnn_dropout,
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Check model_packed.py and these architecture "
            "arguments against the training run: --hidden-dim, --pool-size, "
            "--temporal-dim, --temporal-dropout, --rnn-dropout."
        ) from exc

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Input preprocessing for standalone raw .npy/.npz inference.
# -----------------------------------------------------------------------------
def load_crop_array(path: Path, array_key: str | None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Lip-crop array not found: {path}")
    if path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("--crop must point to a .npy or .npz file.")

    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = array_key or ("arr_0" if "arr_0" in loaded.files else loaded.files[0])
            if key not in loaded.files:
                raise KeyError(
                    f"Array key {key!r} not found. Available keys: {loaded.files}"
                )
            array = loaded[key]
        finally:
            loaded.close()
    else:
        array = loaded

    return np.asarray(array)


def prepare_raw_crop(
    array: np.ndarray,
    normalization: str,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Convert crop array into a model input shaped [1, T, 1, H, W]."""
    if array.ndim == 3:
        # [T, H, W] -> [T, 1, H, W]
        array = array[:, None, :, :]
    elif array.ndim == 4 and array.shape[1] == 1:
        # Already [T, 1, H, W]
        pass
    elif array.ndim == 4 and array.shape[-1] == 1:
        # [T, H, W, 1] -> [T, 1, H, W]
        array = np.transpose(array, (0, 3, 1, 2))
    else:
        raise ValueError(
            "Expected [T,H,W], [T,1,H,W], or [T,H,W,1]; "
            f"got {tuple(array.shape)}."
        )

    if array.shape[0] < 1:
        raise ValueError("Input has zero frames.")
    if array.shape[1] != 1:
        raise ValueError(f"Expected one grayscale channel, got {array.shape}.")

    video = torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32)
    if not torch.isfinite(video).all():
        raise ValueError("Lip crop contains NaN or infinity.")

    if normalization == "zero_one":
        # Correct for uint8 crops or float crops stored on [0, 255].
        if float(video.max()) > 1.5:
            video = video / 255.0
    elif normalization == "zscore":
        if std <= 0:
            raise ValueError("--std must be > 0 for zscore normalization.")
        if float(video.max()) > 1.5:
            video = video / 255.0
        video = (video - mean) / std
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    return video.unsqueeze(0)  # [1, T, 1, H, W]


# -----------------------------------------------------------------------------
# Metrics for CSV evaluation.
# -----------------------------------------------------------------------------
def split_targets(
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    id_to_phone: Dict[int, str],
) -> List[List[str]]:
    references: List[List[str]] = []
    offset = 0
    targets = targets.detach().cpu()

    for length in target_lens.detach().cpu():
        length_int = int(length.item())
        token_ids = targets[offset : offset + length_int].tolist()
        references.append([id_to_phone[token_id] for token_id in token_ids])
        offset += length_int

    return references


def edit_distance_ops(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Return substitutions, deletions, and insertions for one ref/hyp pair."""
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


# -----------------------------------------------------------------------------
# Inference modes.
# -----------------------------------------------------------------------------
@torch.inference_mode()
def infer_one_crop(
    model: TinyLip2PhonemeCTC,
    crop_path: Path,
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    array = load_crop_array(crop_path, args.array_key)
    video = prepare_raw_crop(array, args.normalization, args.mean, args.std)
    input_len = int(video.size(1))

    video = video.to(device, non_blocking=True)
    input_lens = torch.tensor([input_len], dtype=torch.long, device=device)

    log_probs, output_lens = model(video, input_lens)
    output_len = int(output_lens[0].item())
    phone_ids = ctc_prefix_beam_search_single(
        sample_log_probs=log_probs[:output_len, 0, :].detach().cpu(),
        blank_id=args.blank_id,
        beam_width=args.beam_width,
        token_prune=args.token_prune,
    )
    phones = [id_to_phone[phone_id] for phone_id in phone_ids]

    return {
        "input": str(crop_path),
        "input_frames": input_len,
        "ctc_output_frames": output_len,
        "phoneme_ids": phone_ids,
        "phonemes": phones,
        "prediction": " ".join(phones),
    }


@torch.inference_mode()
def infer_metadata_csv(
    model: TinyLip2PhonemeCTC,
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Iterable[Dict[str, Any]], Dict[str, Any]]:
    # Import only in CSV mode. It guarantees identical normalization to train/val.
    from dataset import Lip2PhonemeDataset, collate_lip_phone_batch

    if args.phone_vocab is None:
        raise ValueError("--phone-vocab is required with --metadata-csv.")
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")
    if not args.phone_vocab.is_file():
        raise FileNotFoundError(f"Phone vocabulary not found: {args.phone_vocab}")

    dataset = Lip2PhonemeDataset(
        metadata_csv=str(args.metadata_csv),
        phone_vocab_json=str(args.phone_vocab),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_lip_phone_batch,
        pin_memory=(device.type == "cuda"),
    )

    total_subs = total_dels = total_ins = 0
    total_ref_len = total_hyp_len = 0
    rows: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(loader, start=1):
        video = batch["video"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)

        log_probs, output_lens = model(video, input_lens)
        hypotheses = ctc_beam_decode(
            log_probs=log_probs,
            output_lens=output_lens,
            id_to_phone=id_to_phone,
            blank_id=args.blank_id,
            beam_width=args.beam_width,
            token_prune=args.token_prune,
        )
        references = split_targets(batch["target"], batch["target_lens"], id_to_phone)

        utt_ids = batch.get("utt_ids", [""] * len(references))
        crop_paths = batch.get("crop_paths", [""] * len(references))
        input_lengths = batch["input_lens"].detach().cpu().tolist()
        output_lengths = output_lens.detach().cpu().tolist()

        for utt_id, crop_path, input_frames, ctc_frames, ref, hyp in zip(
            utt_ids,
            crop_paths,
            input_lengths,
            output_lengths,
            references,
            hypotheses,
        ):
            substitutions, deletions, insertions = edit_distance_ops(ref, hyp)
            ref_len = len(ref)
            hyp_len = len(hyp)
            edits = substitutions + deletions + insertions

            rows.append(
                {
                    "utt_id": utt_id,
                    "crop_path": crop_path,
                    "input_frames": int(input_frames),
                    "ctc_output_frames": int(ctc_frames),
                    "ref_len": ref_len,
                    "hyp_len": hyp_len,
                    "per": edits / max(ref_len, 1),
                    "hyp_ref_len_ratio": hyp_len / max(ref_len, 1),
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "reference": " ".join(ref),
                    "prediction": " ".join(hyp),
                }
            )

            total_subs += substitutions
            total_dels += deletions
            total_ins += insertions
            total_ref_len += ref_len
            total_hyp_len += hyp_len

        print(f"Processed batch {batch_index}/{len(loader)}")

    total_edits = total_subs + total_dels + total_ins
    summary = {
        "samples": len(rows),
        "decoder": {
            "type": "prefix_beam_no_lm",
            "beam_width": args.beam_width,
            "token_prune": args.token_prune,
            "blank_id": args.blank_id,
        },
        "corpus_metrics": {
            "reference_phonemes": total_ref_len,
            "predicted_phonemes": total_hyp_len,
            "per": total_edits / max(total_ref_len, 1),
            "sub_rate": total_subs / max(total_ref_len, 1),
            "del_rate": total_dels / max(total_ref_len, 1),
            "ins_rate": total_ins / max(total_ref_len, 1),
            "hyp_ref_len_ratio": total_hyp_len / max(total_ref_len, 1),
        },
    }
    return rows, summary


# -----------------------------------------------------------------------------
# CLI and entry point.
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CTC phoneme inference for TinyLip2PhonemeCTC.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--crop",
        type=Path,
        help="One raw lip crop (.npy/.npz).",
    )
    mode.add_argument(
        "--metadata-csv",
        type=Path,
        help="Infer and score every row using Lip2PhonemeDataset.",
    )

    parser.add_argument(
        "--phone-vocab",
        type=Path,
        default=Path("phone_vocab.json"),
        help="Required in CSV mode; used only as checkpoint fallback in crop mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lip2phoneme_predictions.jsonl"),
    )
    parser.add_argument(
        "--array-key",
        default=None,
        help="Array key for a .npz crop. Defaults to arr_0 or its first array.",
    )

    # Raw crop preprocessing. This must match dataset.py preprocessing.
    parser.add_argument(
        "--normalization",
        choices=["zero_one", "zscore", "none"],
        default="zero_one",
        help="Used only with --crop.",
    )
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)

    parser.add_argument("--blank-id", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--token-prune", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    # Must match the model configuration used in the supplied training script.
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--temporal-dim", type=int, default=192)
    parser.add_argument("--temporal-dropout", type=float, default=0.10)
    parser.add_argument("--rnn-dropout", type=float, default=0.20)

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or e.g. cuda:1",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({value}) but is unavailable.")
    return device


def validate_args(args: argparse.Namespace, id_to_phone: Dict[int, str]) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.blank_id not in id_to_phone:
        raise ValueError(
            f"--blank-id {args.blank_id} is absent from the checkpoint vocabulary."
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    id_to_phone = get_phone_vocab(checkpoint, args.phone_vocab)
    validate_args(args, id_to_phone)
    model = build_model(checkpoint, id_to_phone, device, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    if args.crop is not None:
        row = infer_one_crop(model, args.crop, id_to_phone, device, args)
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Input frames: {row['input_frames']}")
        print(f"CTC output frames: {row['ctc_output_frames']}")
        print(f"Prediction: {row['prediction']}")
        print(f"Saved: {args.output}")
        return

    rows, summary = infer_metadata_csv(model, id_to_phone, device, args)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_val_per": checkpoint.get("val_per"),
            "metadata_csv": str(args.metadata_csv),
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    metrics = summary["corpus_metrics"]
    print(
        "Corpus PER: "
        f"{metrics['per']:.4f} | Sub: {metrics['sub_rate']:.4f} | "
        f"Del: {metrics['del_rate']:.4f} | Ins: {metrics['ins_rate']:.4f} | "
        f"Hyp/Ref: {metrics['hyp_ref_len_ratio']:.4f}"
    )
    print(f"Saved predictions: {args.output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
