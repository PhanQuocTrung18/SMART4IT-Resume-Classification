import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import matplotlib.pyplot as plt

df = pd.read_csv('data_cleaned.csv')

label_col = 'label'
all_labels = sorted(list(set([label for sublist in df[label_col].str.split(';') for label in sublist])))

def multilabel_binarizer(label_str):
    labels = label_str.split(';')
    return [1 if label in labels else 0 for label in all_labels]

y = np.array(df[label_col].apply(multilabel_binarizer).tolist())

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_indices, temp_indices = next(mskf.split(df, y))

train_df = df.iloc[train_indices].reset_index(drop=True)
temp_df = df.iloc[temp_indices].reset_index(drop=True)

y_temp = y[temp_indices]
mskf_temp = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
val_indices, test_indices = next(mskf_temp.split(temp_df, y_temp))

val_df = temp_df.iloc[val_indices].reset_index(drop=True)
test_df = temp_df.iloc[test_indices].reset_index(drop=True)

output_folder = 'data'
os.makedirs(output_folder, exist_ok=True)

train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_folder, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

print(f'Train size: {len(train_df)}')
print(f'Validation size: {len(val_df)}')
print(f'Test size: {len(test_df)}')

def plot_label_distribution_multilabel(df_list, names, label_column, all_labels):
    label_counts = {}
    
    for df, name in zip(df_list, names):
        counts = {label: 0 for label in all_labels}
        for labels in df[label_column]:
            for label in labels.split(';'):
                if label in counts:
                    counts[label] += 1
        label_counts[name] = counts

    labels = all_labels
    x = range(len(labels))
    
    plt.figure(figsize=(16, 6))
    for idx, (name, counts) in enumerate(label_counts.items()):
        plt.bar(
            [p + idx*0.25 for p in x],
            [counts[label] for label in labels],
            width=0.25,
            label=name
        )

    plt.xticks([p + 0.25 for p in x], labels, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('Label Distribution Across Train, Validation, and Test Sets (Multilabel)')
    plt.legend()
    plt.show()

plot_label_distribution_multilabel(
    [train_df, val_df, test_df],
    ['Train', 'Validation', 'Test'],
    label_col,
    all_labels
)

output_folder
