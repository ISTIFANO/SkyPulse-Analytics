"""
Data augmentation utilities for text classification.
Uses nlpaug for synonym-based augmentation to balance classes.
"""

import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw

# Download required NLTK data for augmentation
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def create_synonym_augmenter(aug_p: float = 0.3) -> naw.SynonymAug:
    """
    Create a WordNet synonym augmenter.
    
    Args:
        aug_p: Probability of augmenting each word
        
    Returns:
        SynonymAug augmenter instance
    """
    augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=aug_p)
    return augmenter


def augment_text(text: str, augmenter: naw.SynonymAug, n_aug: int = 1) -> list:
    """
    Augment a single text using the provided augmenter.
    
    Args:
        text: Text to augment
        augmenter: nlpaug augmenter instance
        n_aug: Number of augmented versions to generate
        
    Returns:
        List of augmented texts
    """
    try:
        augmented = augmenter.augment(text, n=n_aug)
        if isinstance(augmented, str):
            return [augmented]
        return augmented
    except Exception:
        return [text] * n_aug


def augment_dataframe(df: pd.DataFrame, text_column: str = "cleaned_text",
                      label_column: str = "label", target_count: int = None,
                      aug_p: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """
    Augment minority classes to balance the dataset.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text to augment
        label_column: Column containing labels
        target_count: Target count per class (default: max class count)
        aug_p: Augmentation probability per word
        random_state: Random seed for reproducibility
        
    Returns:
        Augmented DataFrame with balanced classes
    """
    np.random.seed(random_state)
    
    augmenter = create_synonym_augmenter(aug_p)
    label_counts = df[label_column].value_counts()
    
    if target_count is None:
        target_count = label_counts.max()
    
    augmented_rows = []
    
    for label in label_counts.index:
        label_df = df[df[label_column] == label]
        current_count = len(label_df)
        
        if current_count >= target_count:
            # No augmentation needed
            continue
            
        samples_needed = target_count - current_count
        
        # Sample texts to augment (with replacement if needed)
        sample_indices = np.random.choice(
            label_df.index, 
            size=samples_needed, 
            replace=True
        )
        
        for idx in sample_indices:
            original_text = df.loc[idx, text_column]
            augmented_texts = augment_text(original_text, augmenter, n_aug=1)
            
            for aug_text in augmented_texts:
                new_row = df.loc[idx].copy()
                new_row[text_column] = aug_text
                augmented_rows.append(new_row)
    
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        result_df = pd.concat([df, augmented_df], ignore_index=True)
    else:
        result_df = df.copy()
    
    return result_df


def get_class_distribution(df: pd.DataFrame, label_column: str = "label") -> dict:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        df: Input DataFrame
        label_column: Column containing labels
        
    Returns:
        Dictionary with class counts
    """
    return df[label_column].value_counts().to_dict()
