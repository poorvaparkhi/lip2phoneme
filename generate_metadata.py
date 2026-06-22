import pandas as pd
import numpy as np
from pathlib import Path


def get_textgrid_duration(textgrid_path):
    """
    Reads xmax from the TextGrid header.
    Usually xmax is the total duration in seconds.
    """
    textgrid_path = Path(textgrid_path)

    with open(textgrid_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # First top-level xmax usually gives total duration
            if line.startswith("xmax"):
                # example: xmax = 8.54
                return float(line.split("=")[1].strip())

    raise RuntimeError(f"Could not find xmax in {textgrid_path}")


def generate_metadata_csv(
    lipcrop_npy_dir,
    textgrid_dir,
    output_csv,
    fps=25,
):
    lipcrop_npy_dir = Path(lipcrop_npy_dir)
    textgrid_dir = Path(textgrid_dir)
    output_csv = Path(output_csv)

    rows = []
    missing_textgrids = []

    npy_files = sorted(lipcrop_npy_dir.glob("*.npy"))

    print(f"Found {len(npy_files)} npy files")

    for npy_path in npy_files:
        utt_id = npy_path.stem

        # Find matching TextGrid by same basename
        textgrid_path = textgrid_dir / f"{utt_id}.TextGrid"

        if not textgrid_path.exists():
            # Sometimes extension may be .textgrid
            alt_textgrid_path = textgrid_dir / f"{utt_id}.textgrid"

            if alt_textgrid_path.exists():
                textgrid_path = alt_textgrid_path
            else:
                missing_textgrids.append(str(npy_path))
                continue

        # Get duration from TextGrid
        duration = get_textgrid_duration(textgrid_path)

        # Optional sanity check: compare npy frames with duration * fps
        x = np.load(npy_path, mmap_mode="r")
        num_frames = x.shape[0]
        expected_frames = duration * fps

        rows.append({
            "utt_id": utt_id,
            "crop_path": str(npy_path),
            "textgrid_path": str(textgrid_path),
            "duration": duration,
            "fps": fps,
            "num_frames": num_frames,
            "expected_frames": expected_frames,
        })

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved metadata: {output_csv}")
    print(f"Rows: {len(df)}")
    print(f"Missing TextGrids: {len(missing_textgrids)}")

    if missing_textgrids:
        print("\nMissing TextGrid for these npy files:")
        for path in missing_textgrids[:20]:
            print(path)

        if len(missing_textgrids) > 20:
            print(f"... and {len(missing_textgrids) - 20} more")

    return df


df = generate_metadata_csv(
    lipcrop_npy_dir="/media/newhddd/poorva/lip2phoneme/lip_videos_npys",
    textgrid_dir="/media/newhddd/poorva/hindi_alignments",
    output_csv="/media/newhddd/poorva/lip2phoneme/metadata.csv",
    fps=25,
)

print(df.head())