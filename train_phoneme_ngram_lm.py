#!/usr/bin/env python3
"""
Typical use with the same train.csv / phone_vocab.json as lip2phoneme training:

    python train_phoneme_ngram_lm.py \
        --train-csv train.csv \
        --phone-vocab-json phone_vocab.json \
        --output phoneme_trigram_lm.pt \
        --order 3

Optional faster source when you already have an ID|phones manifest:

    python train_phoneme_ngram_lm.py \
        --phones-manifest train_phones.txt \
        --phone-vocab-json phone_vocab.json \
        --output phoneme_trigram_lm.pt \
        --order 3

The saved .pt file stores the exact n-gram counts. Import `load_phoneme_lm`
from this file in an inference script to recover the LM.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch

LMToken = Union[int, str]


class PhonemeNGramLM:
    """Add-k-smoothed token-ID n-gram LM for CTC shallow fusion.

    It is deliberately external to the lipreading model: it has no torch
    Parameters, no gradients, and no optimizer state.
    """

    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(
        self,
        order: int,
        vocab_ids: Sequence[int],
        add_k: float = 0.10,
    ) -> None:
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}")
        if add_k <= 0:
            raise ValueError(f"add_k must be > 0, got {add_k}")

        self.order = int(order)
        self.context_size = self.order - 1
        self.add_k = float(add_k)

        self.vocab_ids = {int(token_id) for token_id in vocab_ids}
        if not self.vocab_ids:
            raise ValueError("LM vocabulary cannot be empty")

        # EOS is a possible final prediction. BOS appears only in contexts.
        self.prediction_vocab = set(self.vocab_ids)
        self.prediction_vocab.add(self.EOS)

        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.num_sequences = 0
        self.num_tokens = 0

    def fit(self, sequences: Iterable[Sequence[int]]) -> "PhonemeNGramLM":
        """Count n-grams from non-empty train-only phoneme-ID sequences."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.num_sequences = 0
        self.num_tokens = 0

        for sequence in sequences:
            tokens = [int(token_id) for token_id in sequence]
            if not tokens:
                continue

            unknown = set(tokens) - self.vocab_ids
            if unknown:
                raise ValueError(
                    "LM sequence has IDs outside the phone vocabulary: "
                    f"{sorted(unknown)}"
                )

            history: List[LMToken] = [self.BOS] * self.context_size
            for token in tokens + [self.EOS]:
                context = tuple(history[-self.context_size:])
                self.ngram_counts[(context, token)] += 1
                self.context_counts[context] += 1
                history.append(token)

            self.num_sequences += 1
            self.num_tokens += len(tokens)

        if self.num_sequences == 0:
            raise ValueError("No non-empty phoneme sequences were available for LM building")
        return self

    def _context_from_prefix(self, prefix: Sequence[int]) -> Tuple[LMToken, ...]:
        history: List[LMToken] = [self.BOS] * self.context_size
        history.extend(int(token_id) for token_id in prefix)
        return tuple(history[-self.context_size:])

    def log_prob(self, token: LMToken, prefix: Sequence[int]) -> float:
        """Return log P(token | recent collapsed non-blank prefix)."""
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
        """LM score added only for a true new collapsed CTC phone."""
        return self.log_prob(int(token_id), prefix)

    def eos_log_prob(self, prefix: Sequence[int]) -> float:
        """Optional final score log P(<eos> | prefix)."""
        return self.log_prob(self.EOS, prefix)

    def to_checkpoint_state(self) -> Dict:
        """Return a torch-saveable representation with exact learned counts."""
        return {
            "order": self.order,
            "add_k": self.add_k,
            "vocab_ids": sorted(self.vocab_ids),
            "num_sequences": self.num_sequences,
            "num_tokens": self.num_tokens,
            "ngram_counts": [
                {
                    "context": list(context),
                    "token_id": token,
                    "count": int(count),
                }
                for (context, token), count in self.ngram_counts.items()
            ],
            "context_counts": [
                {"context": list(context), "count": int(count)}
                for context, count in self.context_counts.items()
            ],
        }

    @classmethod
    def from_checkpoint_state(cls, state: Dict) -> "PhonemeNGramLM":
        """Restore a previously saved LM state."""
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


def _torch_load(path: Path) -> Dict:
    """Load a local torch artifact across common PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases before weights_only existed.
        return torch.load(path, map_location="cpu")


def load_phoneme_lm(path: Union[str, Path]) -> PhonemeNGramLM:
    """Load an LM saved by this script; usable from an inference script."""
    payload = _torch_load(Path(path))
    if payload.get("format") != "phoneme_ngram_lm_v1":
        raise ValueError(f"Not a supported phoneme LM artifact: {path}")
    return PhonemeNGramLM.from_checkpoint_state(payload["phoneme_lm_state"])


def _read_phone_vocab(vocab_path: Path) -> Dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    return {str(phone): int(token_id) for phone, token_id in raw_vocab.items()}


def _target_ids_from_dataset_sample(sample: Dict, blank_id: int, index: int) -> List[int]:
    """Extract exactly one unpadded CTC target sequence from a dataset sample."""
    if "target" not in sample:
        raise KeyError(
            "Lip2PhonemeDataset did not return a 'target' entry. "
            "This LM builder must receive the same target IDs used by CTC training."
        )

    target = sample["target"]
    if torch.is_tensor(target):
        ids = [int(token_id) for token_id in target.detach().cpu().tolist()]
    else:
        ids = [int(token_id) for token_id in target]

    if not ids:
        raise ValueError(f"Empty target at dataset index {index}")
    if blank_id in ids:
        raise ValueError(
            f"CTC blank ID {blank_id} appeared in train target at index {index}. "
            "Reference targets must never contain the CTC blank."
        )
    return ids


def _read_sequences_from_training_csv(
    train_csv: Path,
    phone_vocab_json: Path,
    blank_id: int,
) -> List[List[int]]:
    """Read targets through the existing dataset class, without any model."""
    try:
        from dataset import Lip2PhonemeDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import Lip2PhonemeDataset from dataset.py. "
            "Run this script from your lip2phoneme project folder, or set PYTHONPATH "
            "so that dataset.py is importable."
        ) from exc

    # This does not instantiate TinyLip2PhonemeCTC or use a GPU.
    dataset = Lip2PhonemeDataset(
        metadata_csv=str(train_csv),
        phone_vocab_json=str(phone_vocab_json),
        augment=False,
    )

    sequences: List[List[int]] = []
    print(f"Reading {len(dataset)} phoneme targets from {train_csv}")
    for index in range(len(dataset)):
        sample = dataset[index]
        sequences.append(_target_ids_from_dataset_sample(sample, blank_id, index))
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f"  targets read: {index + 1}/{len(dataset)}")
    return sequences


def _read_sequences_from_manifest(
    manifest_path: Path,
    phone_to_id: Dict[str, int],
    blank_id: int,
) -> List[List[int]]:
    """Read `utt_id|phone phone ...` lines without loading dataset.py or videos."""
    sequences: List[List[int]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "|" not in line:
                raise ValueError(
                    f"{manifest_path}:{line_number}: expected 'utt_id|phone phone ...'"
                )
            _, phone_text = line.split("|", 1)
            phones = phone_text.split()
            if not phones:
                raise ValueError(f"{manifest_path}:{line_number}: empty phone sequence")

            unknown = [phone for phone in phones if phone not in phone_to_id]
            if unknown:
                raise ValueError(
                    f"{manifest_path}:{line_number}: phones absent from phone_vocab.json: "
                    f"{sorted(set(unknown))}"
                )

            ids = [phone_to_id[phone] for phone in phones]
            if blank_id in ids:
                raise ValueError(
                    f"{manifest_path}:{line_number}: CTC blank appeared in reference phones"
                )
            sequences.append(ids)

    if not sequences:
        raise ValueError(f"No non-empty phone sequences found in {manifest_path}")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train-only phoneme n-gram LM for CTC shallow fusion."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--train-csv",
        help="The same train.csv used by Lip2PhonemeDataset during CTC training.",
    )
    source_group.add_argument(
        "--phones-manifest",
        help="Optional faster source: lines formatted as utt_id|phone phone ...",
    )
    parser.add_argument(
        "--phone-vocab-json",
        required=True,
        help="Exact phone_vocab.json used by the CTC model.",
    )
    parser.add_argument(
        "--output",
        default="phoneme_trigram_lm.pt",
        help="Output .pt LM artifact (default: phoneme_trigram_lm.pt).",
    )
    parser.add_argument("--order", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--add-k", type=float, default=0.10)
    parser.add_argument("--blank-id", type=int, default=0)
    parser.add_argument(
        "--verify-checkpoint",
        default=None,
        help=(
            "Optional CTC checkpoint. Its saved phone_to_id is checked against "
            "--phone-vocab-json before the LM is written."
        ),
    )
    args = parser.parse_args()

    vocab_path = Path(args.phone_vocab_json)
    output_path = Path(args.output)

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Phone vocabulary JSON not found: {vocab_path}")
    if args.add_k <= 0:
        raise ValueError("--add-k must be > 0")

    phone_to_id = _read_phone_vocab(vocab_path)
    if args.blank_id not in phone_to_id.values():
        raise ValueError(
            f"--blank-id {args.blank_id} is not present in {vocab_path}; "
            "use the CTC blank ID from your model."
        )

    if args.verify_checkpoint:
        checkpoint_path = Path(args.verify_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = _torch_load(checkpoint_path)
        checkpoint_vocab = checkpoint.get("phone_to_id")
        if checkpoint_vocab is not None:
            checkpoint_vocab = {
                str(phone): int(token_id)
                for phone, token_id in checkpoint_vocab.items()
            }
            if checkpoint_vocab != phone_to_id:
                raise ValueError(
                    "The supplied checkpoint uses a different phone_to_id mapping. "
                    "Use exactly the phone_vocab.json used when that checkpoint was trained."
                )

    if args.train_csv:
        source_path = Path(args.train_csv)
        if not source_path.is_file():
            raise FileNotFoundError(f"Train CSV not found: {source_path}")
        sequences = _read_sequences_from_training_csv(
            source_path,
            vocab_path,
            args.blank_id,
        )
        source_kind = "train_csv_via_Lip2PhonemeDataset"
    else:
        source_path = Path(args.phones_manifest)
        if not source_path.is_file():
            raise FileNotFoundError(f"Phone manifest not found: {source_path}")
        sequences = _read_sequences_from_manifest(
            source_path,
            phone_to_id,
            args.blank_id,
        )
        source_kind = "phones_manifest"

    vocab_ids = sorted(token_id for token_id in phone_to_id.values() if token_id != args.blank_id)
    lm = PhonemeNGramLM(
        order=args.order,
        vocab_ids=vocab_ids,
        add_k=args.add_k,
    ).fit(sequences)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "phoneme_ngram_lm_v1",
        "source_kind": source_kind,
        "source_path": str(source_path.resolve()),
        "phone_vocab_json": str(vocab_path.resolve()),
        "phone_to_id": phone_to_id,
        "blank_id": int(args.blank_id),
        "phoneme_lm_state": lm.to_checkpoint_state(),
    }
    torch.save(payload, output_path)

    print("\nExternal phoneme LM built successfully")
    print(f"  output: {output_path.resolve()}")
    print(f"  source: {source_kind}")
    print(f"  order: {lm.order}-gram")
    print(f"  smoothing: add_k={lm.add_k}")
    print(f"  train sequences: {lm.num_sequences}")
    print(f"  train phones: {lm.num_tokens}")
    print(f"  unique observed n-grams: {len(lm.ngram_counts)}")
    print(f"  artifact size: {output_path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
