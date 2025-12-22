"""
Data loading utilities for airline sentiment analysis.
Handles loading data from HuggingFace datasets.
"""

from datasets import load_dataset
import pandas as pd


def load_airline_sentiment_data(dataset_name: str = "osanseviero/twitter-airline-sentiment",
                                 split: str = "train") -> pd.DataFrame:
    """
    Load airline sentiment dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (e.g., 'train', 'test')
        
    Returns:
        Pandas DataFrame with the loaded data
    """
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    return df


def explore_dataframe(df: pd.DataFrame) -> dict:
    """
    Perform basic exploratory data analysis on a dataframe.
    
    Args:
        df: Pandas DataFrame to explore
        
    Returns:
        Dictionary with exploration results
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "head": df.head().to_dict()
    }
    return info


def get_label_distribution(df: pd.DataFrame, label_column: str = "airline_sentiment") -> pd.Series:
    """
    Get the distribution of labels in the dataset.
    
    Args:
        df: Pandas DataFrame
        label_column: Name of the label column
        
    Returns:
        Series with label counts
    """
    return df[label_column].value_counts()


def encode_labels(df: pd.DataFrame, label_column: str = "airline_sentiment") -> dict:
    """
    Create label encoding mapping.
    
    Args:
        df: Pandas DataFrame
        label_column: Name of the label column
        
    Returns:
        Dictionary mapping labels to numeric values
    """
    unique_labels = df[label_column].unique()
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    return label_map


def apply_label_encoding(df: pd.DataFrame, label_map: dict, 
                         label_column: str = "airline_sentiment") -> None:
    """
    Apply label encoding to dataframe in-place.
    
    Args:
        df: Pandas DataFrame
        label_map: Dictionary mapping labels to numeric values
        label_column: Name of the label column
    """
    df["label"] = df[label_column].map(label_map)
