import pandas as pd
from sklearn.model_selection import train_test_split

# Loading datasets
train_fp = 'Dataset/train.csv'
test_fp = 'Dataset/test.csv'
test_lbl_fp = 'Dataset/test_labels.csv'

train_df = pd.read_csv(train_fp)
test_df = pd.read_csv(test_fp)
test_lbl = pd.read_csv(test_lbl_fp)

# Merging test.csv with test_labels.csv
test_df = test_df.merge(test_lbl, on='id', how='left')

label_cols = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']

# Removing rows with -1 labels
test_df = test_df[test_df[label_cols].ge(0).all(axis=1)]

# Creating binary toxicity label
for d in [train_df, test_df]:
    d['is_toxic'] = (d[label_cols].sum(axis=1) > 0).astype(int)

print(f"Total samples: {len(train_df)}")
print(f"Toxic samples: {train_df['is_toxic'].sum()} ({train_df['is_toxic'].mean()*100:.2f}%)")
print(f"Non-toxic samples: {(1 - train_df['is_toxic']).sum()} ({(1-train_df['is_toxic'].mean())*100:.2f}%)")

# Train / Validation Split
train_df, val_df = train_test_split(
    train_df[['comment_text', 'is_toxic'] + label_cols],
    test_size=0.1,
    random_state=42,
    stratify=train_df['is_toxic']
)

# Selecting relevant columns for test
test_df = test_df[['comment_text', 'is_toxic'] + label_cols]

# Saving preprocessed data (DO NOT CHANGE THESE NAMES)
train_df.to_csv('Dataset/train_preprocessed.csv', index=False)
val_df.to_csv('Dataset/val_preprocessed.csv', index=False)
test_df.to_csv('Dataset/test_preprocessed.csv', index=False)

print(f"Preprocessed data saved successfully!")
print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")