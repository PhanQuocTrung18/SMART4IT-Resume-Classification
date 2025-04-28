from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Read dataset from csv file
def read_dataset(filepath):
    dataset = pd.read_csv(filepath, on_bad_lines='skip')
    return dataset

def convert_labels_to_one_hot(dataset, label_column, separator=';'):
    dataset[label_column] = dataset[label_column].astype(str).fillna('')
    labels_list = dataset[label_column].apply(lambda x: [label.strip() for label in x.split(separator) if label.strip()])
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels_list)
    one_hot_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)
    return one_hot_df, mlb, labels_list

# Read the datasets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

# Convert labels to one-hot encoding
train_one_hot, mlb, labels_list = convert_labels_to_one_hot(train_df, "label")
val_one_hot, mlb, labels_list = convert_labels_to_one_hot(val_df, "label")
test_one_hot, mlb, labels_list = convert_labels_to_one_hot(test_df, "label")

# Combine text with one-hot labels
train_df = pd.concat([train_df[["text"]].reset_index(drop=True), train_one_hot], axis=1)
val_df = pd.concat([val_df[["text"]].reset_index(drop=True), val_one_hot], axis=1)
test_df = pd.concat([test_df[["text"]].reset_index(drop=True), test_one_hot], axis=1)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Create dataset dictionary
dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
})

# Labels
labels = ['Database_Administrator', 'Front_End_Developer', 'Java_Developer', 'Network_Administrator', 'Project_manager', 'Python_Developer', 'Security_Analyst', 'Software_Developer', 'Systems_Administrator', 'Web_Developer']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Load tokenizer and model (BERT model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

# Preprocessing function
def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)

    # Create numpy array for labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

# TrainingArguments and Trainer
args = TrainingArguments(
    "bert-base-uncased",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

# Compute metrics for multi-label classification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro = f1_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    return {'f1': f1_micro, 'roc_auc': roc_auc, 'accuracy': accuracy}

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(predictions=preds, labels=p.label_ids)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate(encoded_dataset["test"])

# Test on a sample text
text = "I have 3 years experience in working with Python and write softwares for customers."
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}
outputs = trainer.model(**encoding)

logits = outputs.logits
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1

# Turn predicted ids into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)
