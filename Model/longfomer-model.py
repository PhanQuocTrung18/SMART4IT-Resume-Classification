import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Read dataset
def read_dataset(filepath):
    return pd.read_csv(filepath, on_bad_lines='skip')

def convert_labels_to_one_hot(dataset, label_column, separator=';'):
    dataset[label_column] = dataset[label_column].astype(str).fillna('')
    labels_list = dataset[label_column].apply(lambda x: [label.strip() for label in x.split(separator) if label.strip()])
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels_list)
    one_hot_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)
    return one_hot_df, mlb, labels_list

# Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

# One-hot encode labels
train_one_hot, mlb, _ = convert_labels_to_one_hot(train_df, "label")
val_one_hot, _, _ = convert_labels_to_one_hot(val_df, "label")
test_one_hot, _, _ = convert_labels_to_one_hot(test_df, "label")

# Merge text and labels
train_df = pd.concat([train_df[["text"]].reset_index(drop=True), train_one_hot], axis=1)
val_df = pd.concat([val_df[["text"]].reset_index(drop=True), val_one_hot], axis=1)
test_df = pd.concat([test_df[["text"]].reset_index(drop=True), test_one_hot], axis=1)

# Convert to Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Drop index column
for ds in [train_dataset, val_dataset, test_dataset]:
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])

dataset = DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})

labels = mlb.classes_.tolist()
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=1024)
    labels_matrix = np.stack([examples[label] for label in labels], axis=1)
    encoding["labels"] = labels_matrix.tolist()
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset.set_format("torch")

# Calculate pos_weight
label_counts = train_one_hot.sum(axis=0).values
num_samples = len(train_one_hot)
pos_weights = torch.tensor((num_samples - label_counts) / (label_counts + 1e-5), dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

# Custom model with BCEWithLogitsLoss(pos_weight)
class CustomModel(torch.nn.Module):
    def __init__(self, base_model_name, num_labels, pos_weight):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            problem_type="multi_label_classification",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return {"loss": loss, "logits": logits}

model = CustomModel("allenai/longformer-base-4096", len(labels), pos_weights)

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return {
        'f1': f1_score(labels, y_pred, average='micro'),
        'roc_auc': roc_auc_score(labels, y_pred, average='micro'),
        'accuracy': accuracy_score(labels, y_pred)
    }

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(preds, p.label_ids)

training_args = TrainingArguments(
    output_dir="longformer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(encoded_dataset["test"])

# Inference
text = "I have 3 years experience in working with Python and write softwares for customers."
encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024).to(model.model.device)
with torch.no_grad():
    logits = model(**encoding)["logits"]
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze())
predictions = (probs >= 0.5).int().cpu().numpy()
predicted_labels = [id2label[i] for i, val in enumerate(predictions) if val == 1]
print(predicted_labels)