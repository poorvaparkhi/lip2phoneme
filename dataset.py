import json
import unicodedata

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from praatio import textgrid


# ============================================================
# Phone cleaning
# ============================================================

DROP_PHONES = {
    "",
    "sil",
    "sp",
    "spn",
    "SIL",
    "<sil>",

    # standalone modifiers / diacritics
    "ː",
    "ʰ",
    "̤",
    "̥",
    "̃",
    "̩",

    # Devanagari leakage
    "्",
    "़",
    "ॅ",
}

PHONE_REPLACEMENTS = {
    # syllable-like noisy labels -> split into phones
    "sə": ["s", "ə"],
    "pə": ["p", "ə"],
    "lə": ["l", "ə"],
    "t͡ʃə": ["t͡ʃ", "ə"],

    # mixed noisy label
    "nə़": ["n", "ə"],

    # Devanagari leakage -> approximate IPA
    "ऑ": ["ɔ"],
}


def normalize_phone_label(label):
    label = label.strip()
    label = unicodedata.normalize("NFC", label)
    return label


def clean_phone_sequence(raw_phones):
    cleaned = []

    for ph in raw_phones:
        ph = normalize_phone_label(ph)

        if ph in DROP_PHONES:
            continue

        if ph in PHONE_REPLACEMENTS:
            cleaned.extend(PHONE_REPLACEMENTS[ph])
        else:
            cleaned.append(ph)

    return cleaned


# ============================================================
# TextGrid loading
# ============================================================

def load_phone_intervals(textgrid_path, tier_name="phones"):
    tg = textgrid.openTextgrid(
        textgrid_path,
        includeEmptyIntervals=True,
    )

    tier = tg.getTier(tier_name)

    intervals = []

    for start, end, label in tier.entries:
        label = normalize_phone_label(label)
        intervals.append((float(start), float(end), label))

    return intervals


def load_phone_sequence(textgrid_path, tier_name="phones"):
    intervals = load_phone_intervals(textgrid_path, tier_name=tier_name)

    raw_phones = [
        label
        for _, _, label in intervals
    ]

    phones = clean_phone_sequence(raw_phones)

    return phones


# ============================================================
# Lip crop normalization
# ============================================================

def normalize_lipcrops(x):
    """
    x: [T, 96, 96], uint8 or float
    """

    x = x.astype("float32")

    if x.max() > 1.5:
        x = x / 255.0

    mean = x.mean()
    std = x.std() + 1e-6

    x = (x - mean) / std

    return x


# ============================================================
# Dataset
# ============================================================

class Lip2PhonemeDataset(Dataset):
    def __init__(self, metadata_csv, phone_vocab_json):
        self.df = pd.read_csv(metadata_csv)

        with open(phone_vocab_json, "r", encoding="utf-8") as f:
            self.phone_to_id = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        utt_id = row["utt_id"]
        crop_path = row["crop_path"]
        textgrid_path = row["textgrid_path"]

        # Load lip crops: [T, 96, 96]
        video = np.load(crop_path)
        video = normalize_lipcrops(video)

        # Convert to tensor
        video = torch.tensor(video, dtype=torch.float32)

        # Add channel dimension: [T, 1, 96, 96]
        video = video.unsqueeze(1)

        # Load cleaned phone sequence
        phones = load_phone_sequence(textgrid_path)

        target_ids = []

        for p in phones:
            if p not in self.phone_to_id:
                raise KeyError(
                    f"Phone {repr(p)} not found in phone_vocab.json. "
                    f"TextGrid: {textgrid_path}"
                )

            target_ids.append(self.phone_to_id[p])

        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return {
            "utt_id": utt_id,
            "video": video,                 # [T, 1, 96, 96]
            "target": target_ids,           # [N]
            "input_len": video.shape[0],
            "target_len": len(target_ids),
            "phones": phones,
        }


# ============================================================
# Collate function
# ============================================================

def collate_lip_phone_batch(batch):
    batch_size = len(batch)

    utt_ids = [b["utt_id"] for b in batch]
    videos = [b["video"] for b in batch]
    targets = [b["target"] for b in batch]
    

    input_lens = torch.tensor(
        [v.shape[0] for v in videos],
        dtype=torch.long,
    )

    target_lens = torch.tensor(
        [t.shape[0] for t in targets],
        dtype=torch.long,
    )

    max_t = max(v.shape[0] for v in videos)

    padded_videos = torch.zeros(
        batch_size,
        max_t,
        1,
        96,
        96,
        dtype=torch.float32,
    )

    for i, video in enumerate(videos):
        T = video.shape[0]
        padded_videos[i, :T] = video

    targets_concat = torch.cat(targets, dim=0)

    return {
        "utt_ids": utt_ids,
        "video": padded_videos,        # [B, T, 1, 96, 96]
        "target": targets_concat,      # [sum target lengths]
        "input_lens": input_lens,      # [B]
        "target_lens": target_lens,    # [B]
    }



