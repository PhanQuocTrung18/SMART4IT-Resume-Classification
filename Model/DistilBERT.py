import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch

# ==== 1. Read Dataset ====
def read_dataset(filepath):
    dataset = pd.read_csv(filepath, on_bad_lines='skip')
    return dataset

# ==== 2. One-hot encode labels ====
def convert_labels_to_one_hot(dataset, label_column, separator=';'):
    dataset[label_column] = dataset[label_column].astype(str).fillna('')
    labels_list = dataset[label_column].apply(lambda x: [label.strip() for label in x.split(separator) if label.strip()])
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels_list)
    one_hot_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)
    return one_hot_df, mlb, labels_list

# ==== 3. Load Data ====
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

train_one_hot, mlb, _ = convert_labels_to_one_hot(train_df, "label")
val_one_hot, _, _ = convert_labels_to_one_hot(val_df, "label")
test_one_hot, _, _ = convert_labels_to_one_hot(test_df, "label")

train_df = pd.concat([train_df[["text"]].reset_index(drop=True), train_one_hot], axis=1)
val_df = pd.concat([val_df[["text"]].reset_index(drop=True), val_one_hot], axis=1)
test_df = pd.concat([test_df[["text"]].reset_index(drop=True), test_one_hot], axis=1)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

for d in [train_dataset, val_dataset, test_dataset]:
    if "__index_level_0__" in d.column_names:
        d = d.remove_columns(["__index_level_0__"])

dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
})

# ==== 4. Define Labels ====
labels = mlb.classes_.tolist()
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

# ==== 5. Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

# ==== 6. Load DistilBERT Model ====
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ==== 7. Training Setup ====
batch_size = 16
metric_name = "f1"

training_args = TrainingArguments(
    output_dir="distilbert-multi-label",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to="none",
)

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    return {
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'roc_auc': roc_auc_score(y_true, y_pred, average='micro'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(preds, p.label_ids)

# ==== 8. Train Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# ==== 9. Evaluate Model ====
trainer.evaluate(encoded_dataset["test"])

# ==== 10. Inference ====
text = "I have 3 years experience in working with Python and write softwares for customers."
encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
encoding = {k: v.to(device) for k, v in encoding.items()}

with torch.no_grad():
    outputs = model(**encoding)

logits = outputs.logits
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]

print("Predicted labels:", predicted_labels)
