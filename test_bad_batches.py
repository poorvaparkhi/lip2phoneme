import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2


debug_path = Path("bad_batches_debug.jsonl")
video_dir = Path("/media/newhddd/poorva/lip2phoneme/lipvideos")

ASSUMED_FPS = 25.0


# counts per utt_id
utt_counter = Counter()

# counts per utt_id + reason
reason_counter = defaultdict(Counter)


with open(debug_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        reason = row.get("reason", "unknown")

        for utt_id in row["utt_ids"]:
            utt_counter[utt_id] += 1
            reason_counter[utt_id][reason] += 1


def find_video_path(utt_id):
    candidates = [
        video_dir / f"{utt_id}.npy",
        video_dir / f"{utt_id}.npz",
        video_dir / f"{utt_id}.mp4",
        video_dir / f"{utt_id}.avi",
        video_dir / f"{utt_id}.mov",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_video_length(path):
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        frames = arr.shape[0]
        fps = ASSUMED_FPS
        seconds = frames / fps
        return frames, fps, seconds, arr.shape

    if suffix == ".npz":
        data = np.load(path, mmap_mode="r")
        key = list(data.keys())[0]
        arr = data[key]
        frames = arr.shape[0]
        fps = ASSUMED_FPS
        seconds = frames / fps
        return frames, fps, seconds, arr.shape

    if suffix in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            return None, None, None, None

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps is None or fps <= 0:
            fps = ASSUMED_FPS

        seconds = frames / fps
        return frames, fps, seconds, "raw video"

    return None, None, None, None


print(
    f"{'bad_count':>10}  "
    f"{'frames':>8}  "
    f"{'fps':>6}  "
    f"{'seconds':>9}  "
    f"{'mins':>10}  "
    f"{'reasons':<35}  "
    f"utt_id"
)
print("-" * 150)


for utt_id, bad_count in utt_counter.most_common(30):
    video_path = find_video_path(utt_id)

    reasons = dict(reason_counter[utt_id])
    reasons_str = ", ".join(
        f"{reason}:{count}"
        for reason, count in reason_counter[utt_id].most_common()
    )

    if video_path is None:
        print(
            f"{bad_count:>10}  "
            f"{'MISSING':>8}  "
            f"{'-':>6}  "
            f"{'-':>9}  "
            f"{'-':>10}  "
            f"{reasons_str:<35}  "
            f"{utt_id}"
        )
        continue

    frames, fps, seconds, shape_info = get_video_length(video_path)

    if frames is None:
        print(
            f"{bad_count:>10}  "
            f"{'ERROR':>8}  "
            f"{'-':>6}  "
            f"{'-':>9}  "
            f"{'-':>10}  "
            f"{reasons_str:<35}  "
            f"{utt_id}"
        )
        continue

    print(
        f"{bad_count:>10}  "
        f"{frames:>8}  "
        f"{fps:>6.2f}  "
        f"{seconds:>9.2f}  "
        f"{format_time(seconds):>10}  "
        f"{reasons_str:<35}  "
        f"{utt_id}"
    )