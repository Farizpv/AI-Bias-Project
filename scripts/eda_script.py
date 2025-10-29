import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re  # For text cleaning
import nltk  # For advanced cleaning
from nltk.corpus import stopwords
from collections import Counter
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CAPTION_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'results.csv')

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs')

KEYWORDS = ['doctor', 'nurse', 'teacher', 'engineer', 'scientist', 'chef', 'artist', 'builder', 'driver', 'pilot']


# --- Stage 1: Basic Loading, Cleaning, and Statistics ---

def load_data(filepath):
    """Loads the caption data, handling potential delimiter issues."""
    print(f"Attempting to load data from: {filepath}")
    try:
        # skiprows=1 TO SKIP THE HEADER
        df = pd.read_csv(
            filepath,
            delimiter='|',
            header=None,
            names=['image_id', 'comment_number', 'caption'],
            skiprows=1
        )

        df['caption'] = df['caption'].fillna('')

        print("Data loaded successfully. Header skipped and nulls filled.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check the file path and delimiter.")
        return None


def initial_data_stats(df):
    """Calculates and prints basic statistics about the dataframe."""
    if df is None: return
    print("\n--- Initial Data Characteristics ---")
    # We skipped the header, so the total entries will be 1 less
    print(f"Total number of captions (entries): {len(df)}")
    print(f"Number of unique images: {df['image_id'].nunique()}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Info:")
    df.info()


def clean_caption(text):
    """Basic text cleaning: lowercase, remove punctuation, numbers."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()
    return text


def advanced_clean_caption(text):
    """More advanced cleaning: remove stopwords."""
    text = clean_caption(text)
    try:
        # Check if stopwords are available
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords (one-time download)...")
        nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def analyze_caption_length(df):
    """Analyzes and plots the distribution of caption lengths."""
    if df is None: return
    # Create the 'caption_length' column after loading
    df['caption_length'] = df['caption'].apply(lambda x: len(str(x).split()))

    print("\n--- Caption Length Analysis ---")
    print(df['caption_length'].describe())  # Basic stats: min, max, mean, percentiles

    plt.figure(figsize=(10, 6))
    sns.histplot(df['caption_length'], bins=50, kde=True)
    plt.title('Distribution of Caption Lengths (Number of Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    # Save plot to the 'outputs' folder
    save_path = os.path.join(OUTPUT_PATH, 'caption_length_distribution.png')
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    plt.close()


# --- Stage 2: Keyword-Based Analysis ---

def find_keywords(df, keywords):
    """Finds captions containing specific keywords and analyzes frequency."""
    if df is None: return {}
    keyword_counts = {keyword: 0 for keyword in keywords}
    images_with_keywords = {keyword: set() for keyword in keywords}

    # clean before searching
    print("\nApplying advanced cleaning to all captions (this may take a moment)...")
    df['cleaned_caption'] = df['caption'].apply(advanced_clean_caption)
    print("Advanced cleaning complete.")

    print("\n--- Keyword Analysis ---")
    for index, row in df.iterrows():
        caption_words = set(row['cleaned_caption'].split())
        for keyword in keywords:
            if keyword in caption_words:
                keyword_counts[keyword] += 1
                images_with_keywords[keyword].add(row['image_id'])

    # Convert sets to counts of unique images
    unique_image_counts = {keyword: len(images) for keyword, images in images_with_keywords.items()}

    print("Total captions found per keyword:")
    print(keyword_counts)
    print("\nUnique images found per keyword:")
    print(unique_image_counts)

    # Create a DataFrame for plotting
    if not unique_image_counts:
        print("No keywords found, skipping plot.")
        return unique_image_counts

    keyword_df = pd.DataFrame(list(unique_image_counts.items()), columns=['Keyword', 'Unique Image Count'])
    keyword_df = keyword_df.sort_values(by='Unique Image Count', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Keyword', y='Unique Image Count', data=keyword_df)
    plt.title('Number of Unique Images Found per Keyword')
    plt.xlabel('Keyword')
    plt.ylabel('Number of Unique Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot to the 'outputs' folder
    save_path = os.path.join(OUTPUT_PATH, 'keyword_image_counts.png')
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    plt.close()

    return unique_image_counts


if __name__ == "__main__":
    dataframe = load_data(CAPTION_FILE_PATH)
    initial_data_stats(dataframe)
    analyze_caption_length(dataframe)

    keyword_results = find_keywords(dataframe, KEYWORDS)

    print("\n--- EDA Script Finished ---")