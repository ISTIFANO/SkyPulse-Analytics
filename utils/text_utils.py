"""
Text cleaning utilities for airline sentiment analysis.
Contains functions for preprocessing text data.
"""

import re
import unicodedata
import emoji
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Initialize stopwords
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, emojis, URLs, mentions, hashtags, 
    punctuation, extra whitespace, and stopwords.
    
    Args:
        text: Raw text string to clean
        
    Returns:
        Cleaned text string
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove mentions (@user)
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtags (#topic)
    text = re.sub(r"#\w+", "", text)
    
    # Remove punctuation (keep only letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lowercase and remove stopwords
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return " ".join(tokens)


def text_clean(text: str) -> str:
    """
    Alternative text cleaning function with unicode normalization.
    Removes URLs and converts to lowercase.
    
    Args:
        text: Raw text string to clean
        
    Returns:
        Cleaned text string
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def preprocess_dataframe(df, text_column: str = "text", 
                         clean_function=clean_text) -> None:
    """
    Apply text cleaning to a dataframe column in-place.
    
    Args:
        df: Pandas DataFrame
        text_column: Name of column containing text
        clean_function: Cleaning function to apply (default: clean_text)
    """
    df["cleaned_text"] = df[text_column].apply(clean_function)
