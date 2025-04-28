import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("english"))


def clean_html(text):
    """
    Clean and normalize the text content.

    Parameters:
        text (str): Input text.

    Returns:
        str: Cleaned text with HTML tags removed, extra whitespace trimmed,
             and unwanted characters (ASCII question marks and Unicode
             replacement characters) removed.
    """
    try:
        text = str(text) if text is not None else ''
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        text = re.sub(r'[\?\uFFFD]', '', text)
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""


def remove_stopwords(text):
    """
    Remove stopwords from the text using NLTK.

    Parameters:
        text (str): Input text.

    Returns:
        str: Text with stopwords removed.
    """
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
        return " ".join(filtered_words)
    return text


def remove_asterisks(text):
    """
    Remove asterisks from the text.

    Parameters:
        text (str): Input text.

    Returns:
        str: Text with asterisks removed.
    """
    return text.replace('*', '')


def process_chunk(chunk, seen):
    """
    Process a DataFrame chunk:
      - Remove rows with missing 'label'.
      - Clean HTML content in 'text'.
      - Remove stopwords and asterisks.
      - Remove duplicate rows across chunks based on ('text', 'label').

    Parameters:
        chunk (pd.DataFrame): Data chunk.
        seen (set): Set of seen (text, label) pairs.

    Returns:
        pd.DataFrame: Cleaned DataFrame chunk.
    """
    # Remove rows with missing labels
    chunk = chunk.dropna(subset=['label'])
    # Process the 'text' column
    chunk['text'] = chunk['text'].apply(clean_html)
    chunk['text'] = chunk['text'].apply(remove_stopwords)
    chunk['text'] = chunk['text'].apply(remove_asterisks)

    # Remove duplicates across chunks
    mask = chunk.apply(lambda row: (row['text'], row['label']) in seen, axis=1)
    new_chunk = chunk[~mask].copy()

    # Update seen set with new (text, label) pairs
    for _, row in new_chunk.iterrows():
        seen.add((row['text'], row['label']))

    return new_chunk


def remove_duplicates(df):
    """
    Removes duplicate rows based on 'text' and 'label' columns, keeping the first occurrence.
    Prints statistics about the removed rows.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A deduplicated DataFrame.
    """
    num_rows_before = df.shape[0]
    df_deduplicated = df.drop_duplicates(subset=['text', 'label'], keep='first')
    num_rows_after = df_deduplicated.shape[0]

    print(f"Total rows before: {num_rows_before}")
    print(f"Total rows after removing duplicates: {num_rows_after}")
    print(f"Total rows removed: {num_rows_before - num_rows_after}")
    
    return df_deduplicated


def remove_missing_labels(df):
    """
    Removes rows with missing labels from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with 'label' column.
        
    Returns:
        pd.DataFrame: DataFrame with rows containing missing labels removed.
    """
    initial_rows = len(df)
    df_clean = df.dropna(subset=['label'])
    rows_removed = initial_rows - len(df_clean)

    print(f"Initial number of rows: {initial_rows}")
    print(f"Rows with missing labels removed: {rows_removed}")
    print(f"Remaining rows: {len(df_clean)}")
    
    return df_clean


def process_file(input_file, chunksize=10000):
    """
    Process a large CSV file in chunks, accumulate the processed chunks in memory,
    and return the combined DataFrame.
    
    Parameters:
        input_file (str): Path to the input CSV file.
        chunksize (int): Number of rows per chunk.
    
    Returns:
        pd.DataFrame: The combined DataFrame after processing all chunks.
    """
    seen = set()
    chunks_list = []

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        cleaned_chunk = process_chunk(chunk, seen)
        chunks_list.append(cleaned_chunk)
        print(f"Processed a chunk with {len(cleaned_chunk)} rows.")

    # Concatenate all processed chunks
    df_all = pd.concat(chunks_list, ignore_index=True)
    return df_all


if __name__ == "__main__":
    INPUT_FILE = "resumes_data.csv"
    FINAL_OUTPUT_FILE = "data_cleaned.csv"

    # Process the large file in chunks and combine results in memory
    df_final = process_file(INPUT_FILE)

    # Post-processing: Apply additional cleaning steps
    df_final = remove_missing_labels(df_final)
    df_final = remove_duplicates(df_final)

    # Save the final cleaned data (chỉ lưu file cuối cùng)
    df_final.to_csv(FINAL_OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {FINAL_OUTPUT_FILE}")


