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

This dedicated script performs a fixed shallow-fusion decoder grid search in\n--metadata-csv mode whenever --phoneme-lm is supplied. The neural model runs once;\nonly CPU prefix-beam decoding repeats across the fixed LM-weight / insertion-penalty\ngrid defined below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_temporal_1d import TinyLip2PhonemeCTC


# Fixed decoder grid for this dedicated evaluation script.
# Outputs are currently shorter than references, so penalties stop at 0.05.
GRID_LM_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.30)
GRID_INSERTION_PENALTIES = (0.0, 0.025, 0.05)


# -----------------------------------------------------------------------------
# External phoneme n-gram LM + CTC prefix beam search with shallow fusion.
# -----------------------------------------------------------------------------
LMToken = Union[int, str]


class PhonemeNGramLM:
    """
    Read-only add-k-smoothed phoneme n-gram LM saved by train_phoneme_ngram_lm.py.

    This is intentionally not a torch.nn.Module: it has no trainable
    parameters, gradients, optimizer, or GPU work.
    """

    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(
        self,
        order: int,
        vocab_ids: Sequence[int],
        add_k: float,
    ) -> None:
        if order < 2:
            raise ValueError(f"LM order must be >= 2, got {order}")
        if add_k <= 0:
            raise ValueError(f"LM add_k must be > 0, got {add_k}")

        self.order = int(order)
        self.context_size = self.order - 1
        self.add_k = float(add_k)
        self.vocab_ids = {int(token_id) for token_id in vocab_ids}
        if not self.vocab_ids:
            raise ValueError("LM vocabulary cannot be empty")

        # EOS can be predicted only when finalizing a complete phone sequence.
        self.prediction_vocab = set(self.vocab_ids)
        self.prediction_vocab.add(self.EOS)

        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.num_sequences = 0
        self.num_tokens = 0

    @classmethod
    def from_checkpoint_state(cls, state: Dict[str, Any]) -> "PhonemeNGramLM":
        lm = cls(
            order=int(state["order"]),
            vocab_ids=state["vocab_ids"],
            add_k=float(state["add_k"]),
        )
        lm.num_sequences = int(state.get("num_sequences", 0))
        lm.num_tokens = int(state.get("num_tokens", 0))
        lm.ngram_counts = Counter(
            {
                (tuple(item["context"]), item["token_id"]): int(item["count"])
                for item in state["ngram_counts"]
            }
        )
        lm.context_counts = Counter(
            {
                tuple(item["context"]): int(item["count"])
                for item in state["context_counts"]
            }
        )
        return lm

    def _context_from_prefix(self, prefix: Sequence[int]) -> Tuple[LMToken, ...]:
        history: List[LMToken] = [self.BOS] * self.context_size
        history.extend(int(token_id) for token_id in prefix)
        return tuple(history[-self.context_size:])

    def log_prob(self, token: LMToken, prefix: Sequence[int]) -> float:
        """Return log P(token | most recent decoded non-blank phones)."""
        if token == self.BOS:
            raise ValueError("BOS cannot be an LM prediction")
        if token != self.EOS and int(token) not in self.vocab_ids:
            raise ValueError(f"Unknown phone ID for this LM: {token}")

        context = self._context_from_prefix(prefix)
        count = self.ngram_counts[(context, token)]
        context_count = self.context_counts[context]
        probability = (count + self.add_k) / (
            context_count + self.add_k * len(self.prediction_vocab)
        )
        return math.log(probability)

    def extension_log_prob(self, prefix: Sequence[int], token_id: int) -> float:
        """Score one genuine collapsed CTC phone extension."""
        return self.log_prob(int(token_id), prefix)

    def eos_log_prob(self, prefix: Sequence[int]) -> float:
        """Optional end-of-sequence LM score after all phones are decoded."""
        return self.log_prob(self.EOS, prefix)


def torch_load_external_lm(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Phoneme LM artifact not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported phoneme LM artifact: {path}")
    return payload


def load_phoneme_lm(path: Path) -> PhonemeNGramLM:
    """Load a .pt artifact created by train_phoneme_ngram_lm.py."""
    payload = torch_load_external_lm(path)
    if payload.get("format") != "phoneme_ngram_lm_v1":
        raise ValueError(
            f"{path} is not a supported phoneme n-gram LM artifact. "
            "Build it with train_phoneme_ngram_lm.py."
        )
    if "phoneme_lm_state" not in payload:
        raise ValueError(f"{path} has no phoneme_lm_state.")
    return PhonemeNGramLM.from_checkpoint_state(payload["phoneme_lm_state"])


def validate_lm_vocab(
    phoneme_lm: PhonemeNGramLM,
    id_to_phone: Dict[int, str],
    blank_id: int,
) -> None:
    """
    Ensure the external LM and CTC model use the same *non-blank* phone IDs.

    The CTC blank is an alignment-only symbol. It must be in the neural model
    vocabulary but is intentionally excluded from phoneme-LM targets, counts,
    and LM vocabulary.
    """
    if blank_id not in id_to_phone:
        raise ValueError(
            f"CTC blank ID {blank_id} is absent from the model vocabulary."
        )

    model_phone_ids = set(id_to_phone.keys()) - {blank_id}
    if phoneme_lm.vocab_ids != model_phone_ids:
        missing_from_lm = sorted(model_phone_ids - phoneme_lm.vocab_ids)
        extra_in_lm = sorted(phoneme_lm.vocab_ids - model_phone_ids)
        raise ValueError(
            "Phone-ID mismatch between the model checkpoint and phoneme LM "
            "after excluding the CTC blank ID. "
            f"IDs missing from LM: {missing_from_lm}; "
            f"extra IDs in LM: {extra_in_lm}. "
            "Rebuild the LM with the exact phone_vocab.json used for this checkpoint."
        )


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
    phoneme_lm: Optional[PhonemeNGramLM] = None,
    lm_weight: float = 0.0,
    insertion_penalty: float = 0.0,
    lm_score_eos: bool = True,
) -> List[int]:
    """
    Decode one utterance of shape [L, C] into CTC-collapsed phone IDs.

    Beam pruning and final selection use:

        log P_ctc(prefix | frames)
        + lm_weight * log P_lm(prefix)
        - insertion_penalty * len(prefix)

    The LM score and insertion penalty are paid ONLY when a new collapsed
    non-blank phone is appended to a prefix. They are not applied when CTC
    emits a blank or repeats the final phone without an intervening blank.
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
    if lm_weight < 0:
        raise ValueError("lm_weight must be >= 0.")
    if insertion_penalty < 0:
        raise ValueError("insertion_penalty must be >= 0.")
    if not (0 <= blank_id < sample_log_probs.size(1)):
        raise ValueError(
            f"blank_id={blank_id} is invalid for {sample_log_probs.size(1)} classes."
        )

    # prefix -> (log P(prefix ending in blank), log P(prefix ending in nonblank))
    # These remain pure CTC path scores. The external LM is used only to rank
    # complete collapsed prefixes during beam pruning/final selection.
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}
    num_classes = sample_log_probs.size(1)
    top_k = min(token_prune, num_classes)

    # For a fixed collapsed prefix, LM probability is deterministic. Cache it
    # so each prefix's n-gram transitions are computed once.
    lm_prefix_score_cache: Dict[Tuple[int, ...], float] = {(): 0.0}

    def lm_prefix_score(prefix: Tuple[int, ...]) -> float:
        if phoneme_lm is None or lm_weight == 0.0:
            return 0.0
        cached = lm_prefix_score_cache.get(prefix)
        if cached is not None:
            return cached

        parent_prefix = prefix[:-1]
        token_id = prefix[-1]
        score = (
            lm_prefix_score(parent_prefix)
            + phoneme_lm.extension_log_prob(parent_prefix, token_id)
        )
        lm_prefix_score_cache[prefix] = score
        return score

    def fused_rank_score(
        prefix: Tuple[int, ...],
        ctc_state: Tuple[float, float],
        include_eos: bool = False,
    ) -> float:
        p_blank, p_nonblank = ctc_state
        score = logaddexp(p_blank, p_nonblank)

        if phoneme_lm is not None and lm_weight != 0.0:
            lm_score = lm_prefix_score(prefix)
            if include_eos and lm_score_eos:
                lm_score += phoneme_lm.eos_log_prob(prefix)
            score += lm_weight * lm_score

        # This gives every genuinely emitted collapsed phone a fixed cost.
        score -= insertion_penalty * len(prefix)
        return score

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

            # Blank does not create a new output phone: no LM/length cost.
            add_probability(prefix, ends_in_blank=True, score=p_total + blank_logp)

            for token_id in candidate_ids:
                if token_id == blank_id:
                    continue

                token_logp = float(frame[token_id].item())
                last_token = prefix[-1] if prefix else None

                if token_id == last_token:
                    # Same phone with no blank in between stays the same
                    # collapsed prefix: no LM/length cost.
                    add_probability(
                        prefix,
                        ends_in_blank=False,
                        score=p_nonblank + token_logp,
                    )

                    # Same phone after a blank produces a genuine repeated
                    # output phone, so the child prefix gets LM/penalty only
                    # when it is ranked below.
                    repeated_prefix = prefix + (token_id,)
                    add_probability(
                        repeated_prefix,
                        ends_in_blank=False,
                        score=p_blank + token_logp,
                    )
                else:
                    # A distinct phone extends the collapsed prefix.
                    extended_prefix = prefix + (token_id,)
                    add_probability(
                        extended_prefix,
                        ends_in_blank=False,
                        score=p_total + token_logp,
                    )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: fused_rank_score(
                    item[0],
                    item[1],
                    include_eos=False,
                ),
                reverse=True,
            )[:beam_width]
        )

    best_prefix, _ = max(
        beams.items(),
        key=lambda item: fused_rank_score(
            item[0],
            item[1],
            include_eos=True,
        ),
    )
    return list(best_prefix)


def ctc_beam_decode(
    log_probs: torch.Tensor,
    output_lens: torch.Tensor,
    id_to_phone: Dict[int, str],
    blank_id: int,
    beam_width: int,
    token_prune: int,
    phoneme_lm: Optional[PhonemeNGramLM] = None,
    lm_weight: float = 0.0,
    insertion_penalty: float = 0.0,
    lm_score_eos: bool = True,
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
            phoneme_lm=phoneme_lm,
            lm_weight=lm_weight,
            insertion_penalty=insertion_penalty,
            lm_score_eos=lm_score_eos,
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
    phoneme_lm: Optional[PhonemeNGramLM],
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
        phoneme_lm=phoneme_lm,
        lm_weight=args.lm_weight,
        insertion_penalty=args.insertion_penalty,
        lm_score_eos=args.lm_score_eos,
    )
    phones = [id_to_phone[phone_id] for phone_id in phone_ids]

    return {
        "input": str(crop_path),
        "input_frames": input_len,
        "ctc_output_frames": output_len,
        "phoneme_ids": phone_ids,
        "phonemes": phones,
        "prediction": " ".join(phones),
        "decoder": {
            "type": (
                "prefix_beam_shallow_fusion"
                if phoneme_lm is not None
                else "prefix_beam"
            ),
            "beam_width": args.beam_width,
            "token_prune": args.token_prune,
            "blank_id": args.blank_id,
            "phoneme_lm": str(args.phoneme_lm) if phoneme_lm is not None else None,
            "lm_order": phoneme_lm.order if phoneme_lm is not None else None,
            "lm_weight": args.lm_weight,
            "insertion_penalty": args.insertion_penalty,
            "lm_score_eos": args.lm_score_eos if phoneme_lm is not None else None,
        },
    }


@torch.inference_mode()
def infer_metadata_csv(
    model: TinyLip2PhonemeCTC,
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
    phoneme_lm: Optional[PhonemeNGramLM],
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
            phoneme_lm=phoneme_lm,
            lm_weight=args.lm_weight,
            insertion_penalty=args.insertion_penalty,
            lm_score_eos=args.lm_score_eos,
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
            "type": (
                "prefix_beam_shallow_fusion"
                if phoneme_lm is not None
                else "prefix_beam"
            ),
            "beam_width": args.beam_width,
            "token_prune": args.token_prune,
            "blank_id": args.blank_id,
            "phoneme_lm": str(args.phoneme_lm) if phoneme_lm is not None else None,
            "lm_order": phoneme_lm.order if phoneme_lm is not None else None,
            "lm_add_k": phoneme_lm.add_k if phoneme_lm is not None else None,
            "lm_weight": args.lm_weight,
            "insertion_penalty": args.insertion_penalty,
            "lm_score_eos": args.lm_score_eos if phoneme_lm is not None else None,
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
# Efficient decoder grid search.
# -----------------------------------------------------------------------------
def parse_nonnegative_float_grid(value: str, option_name: str) -> List[float]:
    """Parse a comma-separated list such as '0,0.05,0.10'."""
    raw_values = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_values:
        raise ValueError(f"{option_name} must contain at least one numeric value.")

    parsed: List[float] = []
    for raw_value in raw_values:
        try:
            parsed_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value {raw_value!r} in {option_name}. "
                "Use comma-separated non-negative floats."
            ) from exc

        if not math.isfinite(parsed_value) or parsed_value < 0:
            raise ValueError(
                f"Every value in {option_name} must be finite and >= 0; "
                f"got {raw_value!r}."
            )
        if parsed_value not in parsed:
            parsed.append(parsed_value)

    return parsed


def make_decoder_summary(
    args: argparse.Namespace,
    phoneme_lm: Optional[PhonemeNGramLM],
    lm_weight: float,
    insertion_penalty: float,
) -> Dict[str, Any]:
    """Return reproducible metadata for one decoder configuration."""
    use_lm = phoneme_lm is not None and lm_weight > 0.0
    return {
        "type": "prefix_beam_shallow_fusion" if use_lm else "prefix_beam",
        "beam_width": args.beam_width,
        "token_prune": args.token_prune,
        "blank_id": args.blank_id,
        "phoneme_lm": str(args.phoneme_lm) if use_lm else None,
        "lm_order": phoneme_lm.order if use_lm else None,
        "lm_add_k": phoneme_lm.add_k if use_lm else None,
        "lm_weight": float(lm_weight),
        "insertion_penalty": float(insertion_penalty),
        "lm_score_eos": args.lm_score_eos if use_lm else None,
    }


@torch.inference_mode()
def cache_metadata_csv_logits(
    model: TinyLip2PhonemeCTC,
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Run the neural model across metadata_csv exactly once and retain only the
    unpadded CTC log-probabilities required for later decoder sweeps.
    """
    from dataset import Lip2PhonemeDataset, collate_lip_phone_batch

    if args.metadata_csv is None:
        raise ValueError("--metadata-csv is required for --grid-search.")
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

    cached: List[Dict[str, Any]] = []
    cached_values = 0

    for batch_index, batch in enumerate(loader, start=1):
        video = batch["video"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)

        log_probs, output_lens = model(video, input_lens)
        references = split_targets(batch["target"], batch["target_lens"], id_to_phone)

        utt_ids = batch.get("utt_ids", [""] * len(references))
        crop_paths = batch.get("crop_paths", [""] * len(references))
        input_lengths = batch["input_lens"].detach().cpu().tolist()
        output_lengths = output_lens.detach().cpu().tolist()

        for local_index, (
            utt_id,
            crop_path,
            input_frames,
            ctc_frames,
            reference,
        ) in enumerate(
            zip(
                utt_ids,
                crop_paths,
                input_lengths,
                output_lengths,
                references,
            )
        ):
            ctc_frames = int(ctc_frames)
            utterance_log_probs = (
                log_probs[:ctc_frames, local_index, :]
                .detach()
                .cpu()
                .contiguous()
            )
            cached_values += int(utterance_log_probs.numel())

            cached.append(
                {
                    "utt_id": utt_id,
                    "crop_path": crop_path,
                    "input_frames": int(input_frames),
                    "ctc_output_frames": ctc_frames,
                    "reference": reference,
                    "log_probs": utterance_log_probs,
                }
            )

        print(f"Cached CTC logits batch {batch_index}/{len(loader)}")

    cache_mib = cached_values * 4 / (1024 * 1024)
    print(
        f"Cached {len(cached)} utterances for decoder grid search "
        f"({cache_mib:.1f} MiB float32 log-probabilities)."
    )
    return cached


def evaluate_cached_decoder_setting(
    cached_samples: Sequence[Dict[str, Any]],
    id_to_phone: Dict[int, str],
    args: argparse.Namespace,
    phoneme_lm: Optional[PhonemeNGramLM],
    lm_weight: float,
    insertion_penalty: float,
    collect_rows: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decode cached neural outputs for one LM/penalty configuration."""
    use_lm = phoneme_lm if lm_weight > 0.0 else None

    total_subs = total_dels = total_ins = 0
    total_ref_len = total_hyp_len = 0
    rows: List[Dict[str, Any]] = []

    for sample in cached_samples:
        token_ids = ctc_prefix_beam_search_single(
            sample_log_probs=sample["log_probs"],
            blank_id=args.blank_id,
            beam_width=args.beam_width,
            token_prune=args.token_prune,
            phoneme_lm=use_lm,
            lm_weight=lm_weight,
            insertion_penalty=insertion_penalty,
            lm_score_eos=args.lm_score_eos,
        )
        hypothesis = [id_to_phone[token_id] for token_id in token_ids]
        reference = sample["reference"]

        substitutions, deletions, insertions = edit_distance_ops(reference, hypothesis)
        ref_len = len(reference)
        hyp_len = len(hypothesis)

        total_subs += substitutions
        total_dels += deletions
        total_ins += insertions
        total_ref_len += ref_len
        total_hyp_len += hyp_len

        if collect_rows:
            edits = substitutions + deletions + insertions
            rows.append(
                {
                    "utt_id": sample["utt_id"],
                    "crop_path": sample["crop_path"],
                    "input_frames": sample["input_frames"],
                    "ctc_output_frames": sample["ctc_output_frames"],
                    "ref_len": ref_len,
                    "hyp_len": hyp_len,
                    "per": edits / max(ref_len, 1),
                    "hyp_ref_len_ratio": hyp_len / max(ref_len, 1),
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "reference": " ".join(reference),
                    "prediction": " ".join(hypothesis),
                }
            )

    total_edits = total_subs + total_dels + total_ins
    summary = {
        "samples": len(cached_samples),
        "decoder": make_decoder_summary(
            args=args,
            phoneme_lm=phoneme_lm,
            lm_weight=lm_weight,
            insertion_penalty=insertion_penalty,
        ),
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


def write_grid_csv(grid_results: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write compact, sortable grid results for quick shell/spreadsheet review."""
    fieldnames = [
        "rank",
        "lm_weight",
        "insertion_penalty",
        "per",
        "sub_rate",
        "del_rate",
        "ins_rate",
        "hyp_ref_len_ratio",
        "predicted_phonemes",
        "reference_phonemes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid_results:
            writer.writerow({field: row[field] for field in fieldnames})


def run_decoder_grid_search(
    model: TinyLip2PhonemeCTC,
    id_to_phone: Dict[int, str],
    device: torch.device,
    args: argparse.Namespace,
    phoneme_lm: Optional[PhonemeNGramLM],
) -> None:
    """
    Evaluate all requested LM-weight / insertion-penalty combinations.

    Only the best setting's utterance-level predictions are saved to --output.
    Full aggregate metrics for every setting are saved as .grid.json and .grid.csv.
    """
    if args.metadata_csv is None:
        raise ValueError("Grid search requires --metadata-csv.")
    if phoneme_lm is None:
        raise ValueError("Grid search requires --phoneme-lm.")

    lm_weights = list(GRID_LM_WEIGHTS)
    insertion_penalties = list(GRID_INSERTION_PENALTIES)

    cached_samples = cache_metadata_csv_logits(model, id_to_phone, device, args)

    total_settings = len(lm_weights) * len(insertion_penalties)
    grid_results: List[Dict[str, Any]] = []

    setting_index = 0
    for lm_weight in lm_weights:
        for insertion_penalty in insertion_penalties:
            setting_index += 1
            _, summary = evaluate_cached_decoder_setting(
                cached_samples=cached_samples,
                id_to_phone=id_to_phone,
                args=args,
                phoneme_lm=phoneme_lm,
                lm_weight=lm_weight,
                insertion_penalty=insertion_penalty,
                collect_rows=False,
            )
            metrics = summary["corpus_metrics"]
            grid_results.append(
                {
                    "lm_weight": float(lm_weight),
                    "insertion_penalty": float(insertion_penalty),
                    "per": metrics["per"],
                    "sub_rate": metrics["sub_rate"],
                    "del_rate": metrics["del_rate"],
                    "ins_rate": metrics["ins_rate"],
                    "hyp_ref_len_ratio": metrics["hyp_ref_len_ratio"],
                    "predicted_phonemes": metrics["predicted_phonemes"],
                    "reference_phonemes": metrics["reference_phonemes"],
                }
            )
            print(
                f"Grid {setting_index}/{total_settings} | "
                f"LM weight={lm_weight:.4g} | penalty={insertion_penalty:.4g} | "
                f"PER={metrics['per']:.4f} | Sub={metrics['sub_rate']:.4f} | "
                f"Del={metrics['del_rate']:.4f} | Ins={metrics['ins_rate']:.4f} | "
                f"Hyp/Ref={metrics['hyp_ref_len_ratio']:.4f}"
            )

    # PER is the selection criterion. The next keys provide deterministic ties
    # while preferring a more balanced output-length ratio.
    grid_results.sort(
        key=lambda row: (
            row["per"],
            abs(row["hyp_ref_len_ratio"] - 1.0),
            row["del_rate"],
            row["ins_rate"],
        )
    )
    for rank, row in enumerate(grid_results, start=1):
        row["rank"] = rank

    best = grid_results[0]
    print(
        "Best grid setting: "
        f"LM weight={best['lm_weight']:.4g} | "
        f"penalty={best['insertion_penalty']:.4g} | "
        f"PER={best['per']:.4f} | Sub={best['sub_rate']:.4f} | "
        f"Del={best['del_rate']:.4f} | Ins={best['ins_rate']:.4f} | "
        f"Hyp/Ref={best['hyp_ref_len_ratio']:.4f}"
    )

    best_rows, best_summary = evaluate_cached_decoder_setting(
        cached_samples=cached_samples,
        id_to_phone=id_to_phone,
        args=args,
        phoneme_lm=phoneme_lm,
        lm_weight=best["lm_weight"],
        insertion_penalty=best["insertion_penalty"],
        collect_rows=True,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        for row in best_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    best_summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": args.checkpoint_epoch_for_output,
            "checkpoint_val_per": args.checkpoint_val_per_for_output,
            "metadata_csv": str(args.metadata_csv),
            "grid_search": {
                "lm_weights_requested": lm_weights,
                "insertion_penalties_requested": insertion_penalties,
                "num_settings": total_settings,
                "selection_metric": "lowest_per_then_length_balance",
                "best_rank": 1,
                "all_results_sorted_by_per": grid_results,
            },
        }
    )

    summary_path = args.output.with_suffix(".summary.json")
    grid_json_path = args.output.with_suffix(".grid.json")
    grid_csv_path = args.output.with_suffix(".grid.csv")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(best_summary, handle, ensure_ascii=False, indent=2)

    grid_payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": args.checkpoint_epoch_for_output,
        "metadata_csv": str(args.metadata_csv),
        "beam_width": args.beam_width,
        "token_prune": args.token_prune,
        "blank_id": args.blank_id,
        "phoneme_lm": str(args.phoneme_lm) if phoneme_lm is not None else None,
        "lm_order": phoneme_lm.order if phoneme_lm is not None else None,
        "lm_add_k": phoneme_lm.add_k if phoneme_lm is not None else None,
        "lm_score_eos": args.lm_score_eos if phoneme_lm is not None else None,
        "lm_weights_requested": lm_weights,
        "insertion_penalties_requested": insertion_penalties,
        "selection_metric": "lowest_per_then_length_balance",
        "results_sorted_by_per": grid_results,
    }
    with grid_json_path.open("w", encoding="utf-8") as handle:
        json.dump(grid_payload, handle, ensure_ascii=False, indent=2)
    write_grid_csv(grid_results, grid_csv_path)

    print(f"Saved best-setting predictions: {args.output}")
    print(f"Saved best-setting summary: {summary_path}")
    print(f"Saved full grid JSON: {grid_json_path}")
    print(f"Saved full grid CSV: {grid_csv_path}")


# -----------------------------------------------------------------------------
# CLI and entry point.
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="CTC phoneme inference and fixed decoder grid search for TinyLip2PhonemeCTC.",
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
    parser.add_argument(
        "--phoneme-lm",
        type=Path,
        default=None,
        help=(
            "Optional .pt n-gram LM built by train_phoneme_ngram_lm.py. "
            "When supplied, shallow fusion is used in CTC prefix beam search."
        ),
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=0.0,
        help="Shallow-fusion weight alpha for external LM log probabilities.",
    )
    parser.add_argument(
        "--insertion-penalty",
        type=float,
        default=0.0,
        help="Fixed cost beta subtracted for every emitted collapsed phone.",
    )
    parser.add_argument(
        "--no-lm-score-eos",
        dest="lm_score_eos",
        action="store_false",
        help="Do not add the external LM's end-of-sequence score at final selection.",
    )
    parser.set_defaults(lm_score_eos=True)
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
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1.")
    if args.token_prune < 1:
        raise ValueError("--token-prune must be >= 1.")
    if args.lm_weight < 0:
        raise ValueError("--lm-weight must be >= 0.")
    if args.insertion_penalty < 0:
        raise ValueError("--insertion-penalty must be >= 0.")
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

    phoneme_lm: Optional[PhonemeNGramLM] = None
    if args.phoneme_lm is not None:
        phoneme_lm = load_phoneme_lm(args.phoneme_lm)
        validate_lm_vocab(phoneme_lm, id_to_phone, args.blank_id)

    model = build_model(checkpoint, id_to_phone, device, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Keep checkpoint metadata available to the grid-search output writer.
    args.checkpoint_epoch_for_output = checkpoint.get("epoch")
    args.checkpoint_val_per_for_output = checkpoint.get("val_per")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # This dedicated grid script automatically runs the fixed grid whenever
    # metadata evaluation and an external LM are supplied. No grid CLI flags.
    if args.metadata_csv is not None and phoneme_lm is not None:
        print(
            "Decoder grid search: model forward pass will run once; "
            "only CPU prefix-beam decoding repeats per grid setting."
        )
        print(f"Fixed grid LM weights: {list(GRID_LM_WEIGHTS)}")
        print(f"Fixed grid insertion penalties: {list(GRID_INSERTION_PENALTIES)}")
        run_decoder_grid_search(
            model=model,
            id_to_phone=id_to_phone,
            device=device,
            args=args,
            phoneme_lm=phoneme_lm,
        )
        return

    if phoneme_lm is None:
        print(
            "Decoder: prefix beam "
            f"(beam={args.beam_width}, token_prune={args.token_prune}, no LM, "
            f"insertion_penalty={args.insertion_penalty})"
        )
    else:
        print(
            "Decoder: shallow-fusion prefix beam "
            f"(LM={args.phoneme_lm}, order={phoneme_lm.order}, "
            f"weight={args.lm_weight}, insertion_penalty={args.insertion_penalty}, "
            f"beam={args.beam_width}, token_prune={args.token_prune}, "
            f"score_eos={args.lm_score_eos})"
        )

    if args.crop is not None:
        row = infer_one_crop(
            model,
            args.crop,
            id_to_phone,
            device,
            args,
            phoneme_lm,
        )
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Input frames: {row['input_frames']}")
        print(f"CTC output frames: {row['ctc_output_frames']}")
        print(f"Prediction: {row['prediction']}")
        print(f"Saved: {args.output}")
        return

    rows, summary = infer_metadata_csv(
        model,
        id_to_phone,
        device,
        args,
        phoneme_lm,
    )
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
