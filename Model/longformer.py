import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# Read dataset from csv file
def read_dataset(filepath):
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset = pd.read_csv(filepath, on_bad_lines='skip')
    return dataset

def convert_labels_to_one_hot(dataset, label_column, separator=';'):
    """
    Converts a multi-label string column into one-hot encoded format.

    Args:
        dataset (pd.DataFrame): The dataset containing a multi-label column.
        label_column (str): Name of the label column.
        separator (str): Separator used between labels in the string.

    Returns:
        pd.DataFrame: One-hot encoded labels as a DataFrame.
        MultiLabelBinarizer: The fitted binarizer for inverse transformations if needed.
    """
    # Ensure all values are strings, replace NaN with an empty string
    dataset[label_column] = dataset[label_column].astype(str).fillna('')

    # Split the labels by separator and strip whitespace
    labels_list = dataset[label_column].apply(lambda x: [label.strip() for label in x.split(separator) if label.strip()])

    # Initialize and fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels_list)

    # Create DataFrame for one-hot labels
    one_hot_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)

    return one_hot_df, mlb, labels_list

# Read the three datasets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
print(train_df.head())

# Convert labels to one-hot encoding (assuming convert_labels_to_one_hot is defined)
train_one_hot, mlb, labels_list = convert_labels_to_one_hot(train_df, "label")
val_one_hot, mlb, labels_list = convert_labels_to_one_hot(val_df, "label")
test_one_hot, mlb, labels_list = convert_labels_to_one_hot(test_df, "label")

# Combine text with one-hot labels
train_df = pd.concat([train_df[["text"]].reset_index(drop=True), train_one_hot], axis=1)
val_df = pd.concat([val_df[["text"]].reset_index(drop=True), val_one_hot], axis=1)
test_df = pd.concat([test_df[["text"]].reset_index(drop=True), test_one_hot], axis=1)
train_df.head()

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Remove unnecessary columns if present
for dataset in [train_dataset, val_dataset, test_dataset]:
    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

# Create dataset dictionary
dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
})

labels = ['Database_Administrator', 'Front_End_Developer', 'Java_Developer', 'Network_Administrator', 'Project_manager', 'Python_Developer', 'Security_Analyst', 'Software_Developer', 'Systems_Administrator', 'Web_Developer']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Load Longformer tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def preprocess_data(examples):
    # Take a batch of texts
    text = examples["text"]

    # Encode them using Longformer tokenizer
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=1024)  # Adjust max_length if needed

    # Add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}

    # Create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))

    # Fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

encoded_dataset.set_format("torch")

# "allenai/longformer-large-4096"
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"longformer",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to="none",
    #push_to_hub=True,
)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0).to(device),
                labels=encoded_dataset['train'][0]['labels'].unsqueeze(0).to(device))

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate(encoded_dataset["test"])

text = "I have 3 years experience in working with Python and write softwares for customers."
print(text)

encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding)

logits = outputs.logits
print(logits.shape)

# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)