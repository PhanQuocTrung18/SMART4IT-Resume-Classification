import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load data ===
def convert_labels_to_one_hot(dataset, label_column, separator=';'):
    dataset[label_column] = dataset[label_column].astype(str).fillna('')
    labels_list = dataset[label_column].apply(lambda x: [label.strip() for label in x.split(separator) if label.strip()])
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels_list)
    one_hot_df = pd.DataFrame(one_hot_labels, columns=mlb.classes_)
    return one_hot_df, mlb, labels_list

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

for ds in [train_dataset, val_dataset, test_dataset]:
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])

dataset = DatasetDict({
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
})

labels = mlb.classes_.tolist()
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

# === Tokenizer ===
model_ckpt = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def preprocess_data(examples):
    text = examples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    labels_batch = {k: examples[k] for k in examples if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
encoded_dataset.set_format("torch")

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    problem_type="multi_label_classification",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# === Training setup ===
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

batch_size = 8

training_args = TrainingArguments(
    output_dir="electra-base-multilabel",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=True
)

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    f1_micro = f1_score(labels, y_pred, average='micro')
    roc_auc = roc_auc_score(labels, y_pred, average='micro')
    acc = accuracy_score(labels, y_pred)
    return {"f1": f1_micro, "roc_auc": roc_auc, "accuracy": acc}

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(preds, p.label_ids)

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
trainer.evaluate(encoded_dataset["test"])

# === Inference Example ===
text = "Experienced in data science and building machine learning models using Python."
encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
encoding = {k: v.to(model.device) for k, v in encoding.items()}

outputs = model(**encoding)
logits = outputs.logits
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())

predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
predicted_labels = [id2label[i] for i, val in enumerate(predictions) if val == 1.0]
print(predicted_labels)
