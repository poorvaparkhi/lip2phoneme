import pandas as pd
from sklearn.model_selection import train_test_split

# input metadata
metadata_csv = "/media/newhddd/poorva/lip2phoneme/metadata.csv"

# output files
train_csv = "/media/newhddd/poorva/lip2phoneme/train.csv"
val_csv = "/media/newhddd/poorva/lip2phoneme/val.csv"

# split settings
val_ratio = 0.05
seed = 42

# load metadata
df = pd.read_csv(metadata_csv)

# shuffle + split
train_df, val_df = train_test_split(
    df,
    test_size=val_ratio,
    random_state=seed,
    shuffle=True,
)

# save
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"Total samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Saved: {train_csv}, {val_csv}")