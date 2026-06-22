import pandas as pd
import numpy as np
from dataset import load_phone_sequence

def check_ctc_lengths(csv_path):
    df = pd.read_csv(csv_path)

    bad = []
    good = []

    for _, row in df.iterrows():
        utt_id = row["utt_id"]
        crop_path = row["crop_path"]
        textgrid_path = row["textgrid_path"]

        video = np.load(crop_path)
        phones = load_phone_sequence(textgrid_path)

        T = video.shape[0]
        N = len(phones)

        if T < N:
            bad.append({
                "utt_id": utt_id,
                "crop_path": crop_path,
                "textgrid_path": textgrid_path,
                "frames": T,
                "phones": N,
                "ratio": T / max(N, 1),
            })
        else:
            good.append({
                "utt_id": utt_id,
                "crop_path": crop_path,
                "textgrid_path": textgrid_path,
                "frames": T,
                "phones": N,
                "ratio": T / max(N, 1),
            })

    bad_df = pd.DataFrame(bad)
    good_df = pd.DataFrame(good)

    print("Total samples:", len(df))
    print("Good samples:", len(good_df))
    print("Bad samples:", len(bad_df))

    if len(bad_df) > 0:
        print(bad_df.sort_values("ratio").head(20))

    return good_df, bad_df


good_df, bad_df = check_ctc_lengths("train.csv")