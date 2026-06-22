import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def mp4_to_lipcrop_npy(
    input_mp4,
    output_dir,
    size=96,
    target_fps=None,
    overwrite=False,
):
    input_mp4 = Path(input_mp4)
    output_dir = Path(output_dir)

    output_npy = output_dir / f"{input_mp4.stem}.npy"

    if output_npy.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_npy}")
        return output_npy

    cap = cv2.VideoCapture(str(input_mp4))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_mp4}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps is not None:
        if original_fps <= 0:
            raise RuntimeError(
                f"Could not read original FPS from video: {input_mp4}"
            )

        frame_step = max(1, round(original_fps / target_fps))
    else:
        frame_step = 1

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(
                gray,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )

            frames.append(gray)

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {input_mp4}")

    arr = np.stack(frames, axis=0).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, arr)

    print(
        f"[OK] {input_mp4.name} -> {output_npy.name} | "
        f"orig_fps={original_fps:.2f}, orig_frames={frame_count}, "
        f"saved_shape={arr.shape}"
    )

    return output_npy


def find_mp4s(input_dir, recursive=False):
    input_dir = Path(input_dir)

    if recursive:
        mp4s = sorted(input_dir.rglob("*.mp4"))
    else:
        mp4s = sorted(input_dir.glob("*.mp4"))

    return mp4s


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input-mp4",
        help="Path to one input mp4",
    )

    group.add_argument(
        "--input-dir",
        help="Directory containing mp4 files",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where .npy files will be saved",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Output frame size. Default: 96",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional target FPS, e.g. 25",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input-dir recursively for mp4 files",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_mp4:
        mp4_to_lipcrop_npy(
            input_mp4=args.input_mp4,
            output_dir=output_dir,
            size=args.size,
            target_fps=args.target_fps,
            overwrite=args.overwrite,
        )
    else:
        mp4_paths = find_mp4s(
            input_dir=args.input_dir,
            recursive=args.recursive,
        )

        print(f"Found {len(mp4_paths)} mp4 files")

        if len(mp4_paths) == 0:
            return

        failed = []

        for mp4_path in tqdm(mp4_paths):
            try:
                mp4_to_lipcrop_npy(
                    input_mp4=mp4_path,
                    output_dir=output_dir,
                    size=args.size,
                    target_fps=args.target_fps,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                print(f"[FAIL] {mp4_path}: {e}")
                failed.append((str(mp4_path), str(e)))

        print("\nDone.")
        print(f"Successful: {len(mp4_paths) - len(failed)}")
        print(f"Failed: {len(failed)}")

        if failed:
            failed_log = output_dir / "failed_mp4_to_npy.txt"

            with open(failed_log, "w", encoding="utf-8") as f:
                for path, err in failed:
                    f.write(f"{path}\t{err}\n")

            print(f"Failed log saved to: {failed_log}")


if __name__ == "__main__":
    main()