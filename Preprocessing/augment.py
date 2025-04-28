# import pandas as pd
# import random
# import re
# import nltk
# import spacy
# from nltk.corpus import wordnet
# from tqdm import tqdm

# # Nếu chưa download WordNet và Spacy
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nlp = spacy.load("en_core_web_sm")

# # =========================
# # 1. PARAMETER SETUP
# # =========================
# MAX_TOKENS = 1024
# NUM_SHUFFLE_COPIES = 1
# NUM_EDA_COPIES = 1
# TOTAL_AUG_COPIES = NUM_SHUFFLE_COPIES + NUM_EDA_COPIES

# # =========================
# # 2. TOKENIZER UTILITIES 
# # =========================

# def regex_tokenizer(text):
#     tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
#     return tokens

# def count_tokens(text):
#     tokens = regex_tokenizer(text)
#     return len(tokens)

# # =========================
# # 3. SHUFFLE SENTENCES
# # =========================

# def spacy_sentence_split(text):
#     doc = nlp(text)
#     sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
#     return sentences

# def shuffle_sentences(text):
#     sentences = spacy_sentence_split(text)
    
#     if len(sentences) <= 1:
#         return text

#     random.shuffle(sentences)
#     return ' '.join(sentences)

# def should_shuffle(text, max_tokens=MAX_TOKENS):
#     token_count = count_tokens(text)
#     return token_count <= max_tokens

# # =========================
# # 4. EDA UTILITIES
# # =========================

# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonym = lemma.name().replace('_', ' ').lower()
#             if synonym != word:
#                 synonyms.add(synonym)
#     return list(synonyms)

# def synonym_replacement(sentence, n):
#     words = sentence.split()
#     new_words = words.copy()
#     candidates = [word for word in words if get_synonyms(word)]
#     if len(candidates) == 0:
#         return sentence
#     n = min(n, len(candidates))
#     random_words = random.sample(candidates, n)

#     for random_word in random_words:
#         synonyms = get_synonyms(random_word)
#         if synonyms:
#             synonym = random.choice(synonyms)
#             new_words = [synonym if word == random_word else word for word in new_words]
#     return ' '.join(new_words)

# def eda(sentence, alpha_sr=0.1, num_aug=4):
#     sentence = sentence.lower()
#     words = sentence.split()
#     num_words = len(words)

#     augmented_sentences = []
#     n_sr = max(1, int(alpha_sr * num_words))
#     augmented_sentences.append(synonym_replacement(sentence, n_sr))

#     while len(augmented_sentences) < num_aug:
#         augmented_sentences.append(random.choice(augmented_sentences))

#     return augmented_sentences

# # =========================
# # 5. FULL AUGMENTATION PIPELINE
# # =========================

# def augment_row(row, shuffle_copies=1, eda_copies=2):
#     augmented_rows = []
#     orig_text = row["text"]
#     orig_token_count = count_tokens(orig_text)

#     if should_shuffle(orig_text, max_tokens=MAX_TOKENS):
#         for _ in range(shuffle_copies):
#             shuffled_text = shuffle_sentences(orig_text)
#             shuffled_token_count = count_tokens(shuffled_text)
#             if abs(orig_token_count - shuffled_token_count) > 50:
#                 continue
#             new_row = row.copy()
#             new_row["text"] = shuffled_text
#             augmented_rows.append(new_row)

#     eda_texts = eda(orig_text, num_aug=eda_copies)
#     for eda_text in eda_texts:
#         eda_token_count = count_tokens(eda_text)
#         if abs(orig_token_count - eda_token_count) > 50:
#             continue
#         new_row = row.copy()
#         new_row["text"] = eda_text
#         augmented_rows.append(new_row)

#     return augmented_rows

# def augment_dataframe(df, shuffle_copies=1, eda_copies=2):
#     augmented_rows = []
#     next_id = df["file_id"].max() + 1 if "file_id" in df.columns else len(df) + 1

#     print("Augmenting data...")
#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         if "Software_Developer" in row["label"]:
#             continue  # Bỏ qua nếu nhãn chứa "Software_Developer"
#         row_augments = augment_row(row, shuffle_copies, eda_copies)
#         for new_row in row_augments:
#             new_row["file_id"] = next_id
#             next_id += 1
#             augmented_rows.append(new_row)

#     augmented_df = pd.DataFrame(augmented_rows)
#     combined_df = pd.concat([df, augmented_df], ignore_index=True)
#     return combined_df

# # =========================
# # 6. RUN PIPELINE
# # =========================

# df = pd.read_csv("data_cleaned.csv")

# # Augment dữ liệu bằng shuffle câu + EDA
# augmented_df = augment_dataframe(df, shuffle_copies=NUM_SHUFFLE_COPIES, eda_copies=NUM_EDA_COPIES)

# # Lưu file augmented
# augmented_df.to_csv("datafinal.csv", index=False)

# print(f"Dữ liệu sau augment: {len(augmented_df)}")
# print(augmented_df.head())

import os
import pandas as pd
import random
import re
import nltk
import spacy
from nltk.corpus import wordnet
from tqdm import tqdm

# =========================
# 1. DOWNLOAD & LOAD NLP TOOLS
# =========================
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")

# =========================
# 2. PARAMETER SETUP
# =========================
MAX_TOKENS = 1024
NUM_SHUFFLE_COPIES = 1
NUM_EDA_COPIES = 1
TOTAL_AUG_COPIES = NUM_SHUFFLE_COPIES + NUM_EDA_COPIES

# =========================
# 3. TOKENIZER UTILITIES 
# =========================
def regex_tokenizer(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def count_tokens(text):
    tokens = regex_tokenizer(text)
    return len(tokens)

# =========================
# 4. SHUFFLE SENTENCES
# =========================
def spacy_sentence_split(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def shuffle_sentences(text):
    sentences = spacy_sentence_split(text)
    
    if len(sentences) <= 1:
        return text

    random.shuffle(sentences)
    return ' '.join(sentences)

def should_shuffle(text, max_tokens=MAX_TOKENS):
    token_count = count_tokens(text)
    return token_count <= max_tokens

# =========================
# 5. EDA UTILITIES
# =========================
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    candidates = [word for word in words if get_synonyms(word)]
    
    if len(candidates) == 0:
        return sentence

    n = min(n, len(candidates))
    random_words = random.sample(candidates, n)

    for random_word in random_words:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
    return ' '.join(new_words)

def eda(sentence, alpha_sr=0.1, num_aug=4):
    sentence = sentence.lower()
    words = sentence.split()
    num_words = len(words)

    augmented_sentences = []
    n_sr = max(1, int(alpha_sr * num_words))
    augmented_sentences.append(synonym_replacement(sentence, n_sr))

    while len(augmented_sentences) < num_aug:
        augmented_sentences.append(random.choice(augmented_sentences))

    return augmented_sentences

# =========================
# 6. FULL AUGMENTATION PIPELINE
# =========================
def augment_row(row, shuffle_copies=1, eda_copies=2):
    augmented_rows = []
    orig_text = row["text"]
    orig_token_count = count_tokens(orig_text)

    # Shuffle sentences
    if should_shuffle(orig_text, max_tokens=MAX_TOKENS):
        for _ in range(shuffle_copies):
            shuffled_text = shuffle_sentences(orig_text)
            shuffled_token_count = count_tokens(shuffled_text)
            if abs(orig_token_count - shuffled_token_count) > 50:
                continue
            new_row = row.copy()
            new_row["text"] = shuffled_text
            augmented_rows.append(new_row)

    # EDA
    eda_texts = eda(orig_text, num_aug=eda_copies)
    for eda_text in eda_texts:
        eda_token_count = count_tokens(eda_text)
        if abs(orig_token_count - eda_token_count) > 50:
            continue
        new_row = row.copy()
        new_row["text"] = eda_text
        augmented_rows.append(new_row)

    return augmented_rows

def augment_dataframe(df, shuffle_copies=1, eda_copies=2):
    augmented_rows = []
    next_id = df["file_id"].max() + 1 if "file_id" in df.columns else len(df) + 1

    print("Augmenting data...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if "Software_Developer" in row["label"]:
            continue  # Bỏ qua nếu nhãn chứa "Software_Developer"
        row_augments = augment_row(row, shuffle_copies, eda_copies)
        for new_row in row_augments:
            new_row["file_id"] = next_id
            next_id += 1
            augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    return combined_df

# =========================
# 7. RUN PIPELINE FOR MULTIPLE FILES
# =========================
def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = ["train.csv", "val.csv", "test.csv"]

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"aug_{file_name}")

        print(f"\nProcessing file: {file_name}")
        df = pd.read_csv(input_path)

        augmented_df = augment_dataframe(df, shuffle_copies=NUM_SHUFFLE_COPIES, eda_copies=NUM_EDA_COPIES)

        augmented_df.to_csv(output_path, index=False)
        print(f"File saved: {output_path} (total rows: {len(augmented_df)})")

# =========================
# 8. MAIN FUNCTION
# =========================
if __name__ == "__main__":
    input_folder = "data"
    output_folder = "data_augmented"

    process_files(input_folder, output_folder)

