"""
ChromaDB utilities for vector storage and retrieval.
Handles storing and loading embeddings with metadata.
"""

import chromadb
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


def create_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    """
    Create a ChromaDB client.
    
    Args:
        persist_directory: Path for persistent storage (None for in-memory)
        
    Returns:
        ChromaDB client instance
    """
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()
    
    return client


def get_or_create_collection(client: chromadb.Client, 
                              collection_name: str) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection
    """
    return client.get_or_create_collection(name=collection_name)


def add_in_batches(collection: chromadb.Collection,
                   ids: List[str],
                   embeddings: List[List[float]],
                   metadatas: List[Dict[str, Any]],
                   documents: List[str],
                   batch_size: int = 1000) -> None:
    """
    Add documents to a ChromaDB collection in batches.
    
    Args:
        collection: ChromaDB collection
        ids: List of document IDs
        embeddings: List of embedding vectors
        metadatas: List of metadata dictionaries
        documents: List of document texts
        batch_size: Number of documents per batch
    """
    total = len(ids)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end]
        )
        
        print(f"Added batch {start//batch_size + 1}: {start} to {end} ({end}/{total})")


def prepare_chromadb_data(df: pd.DataFrame,
                          id_prefix: str,
                          text_column: str = "cleaned_text",
                          label_column: str = "label",
                          embedding_column: str = "embedding") -> tuple:
    """
    Prepare data for ChromaDB insertion from a DataFrame.
    
    Args:
        df: DataFrame with text, labels, and embeddings
        id_prefix: Prefix for document IDs
        text_column: Column containing text
        label_column: Column containing labels
        embedding_column: Column containing embeddings
        
    Returns:
        Tuple of (ids, embeddings, metadatas, documents)
    """
    ids = [f"{id_prefix}_{i}" for i in range(len(df))]
    
    embeddings = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                  for emb in df[embedding_column]]
    
    metadatas = [{"label": int(label)} for label in df[label_column]]
    
    documents = df[text_column].tolist()
    
    return ids, embeddings, metadatas, documents


def get_collection_data(collection: chromadb.Collection,
                        include: List[str] = None) -> Dict[str, Any]:
    """
    Retrieve all data from a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        include: List of fields to include (default: embeddings, metadatas)
        
    Returns:
        Dictionary with collection data
    """
    if include is None:
        include = ["embeddings", "metadatas"]
    
    return collection.get(include=include)


def load_train_test_data(client: chromadb.Client,
                         train_collection_name: str = "my_train_collection",
                         test_collection_name: str = "my_test_collection") -> tuple:
    """
    Load train and test data from ChromaDB collections.
    
    Args:
        client: ChromaDB client
        train_collection_name: Name of training collection
        test_collection_name: Name of test collection
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    train_collection = client.get_or_create_collection(train_collection_name)
    test_collection = client.get_or_create_collection(test_collection_name)
    
    train_data = train_collection.get(include=["embeddings", "metadatas"])
    test_data = test_collection.get(include=["embeddings", "metadatas"])
    
    # Convert to numpy arrays
    X_train = np.array([np.array(e, dtype=float) for e in train_data["embeddings"]])
    y_train = np.array([m["label"] for m in train_data["metadatas"]])
    
    X_test = np.array([np.array(e, dtype=float) for e in test_data["embeddings"]])
    y_test = np.array([m["label"] for m in test_data["metadatas"]])
    
    return X_train, y_train, X_test, y_test


def delete_collection(client: chromadb.Client, collection_name: str) -> None:
    """
    Delete a ChromaDB collection.
    
    Args:
        client: ChromaDB client
        collection_name: Name of collection to delete
    """
    client.delete_collection(name=collection_name)
