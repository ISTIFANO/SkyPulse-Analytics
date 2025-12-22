"""
SkyPulse Analytics - Batch Processing Module

This module contains utilities for:
- Data loading (data_loader.py)
- Text preprocessing and augmentation (augmentation.py) 
- Embedding generation (embedding.py)
- ChromaDB vector storage (chromadb_utils.py)
- Model training (train.py)
- Model evaluation (evaluate.py)
"""

from .data_loader import (
    load_airline_sentiment_data,
    explore_dataframe,
    get_label_distribution,
    encode_labels,
    apply_label_encoding
)

from .augmentation import (
    create_synonym_augmenter,
    augment_text,
    augment_dataframe,
    get_class_distribution
)

from .embedding import (
    load_embedding_model,
    generate_embeddings,
    add_embeddings_to_dataframe,
    to_2d_array,
    get_embedding_dimension
)

from .chromadb_utils import (
    create_client,
    get_or_create_collection,
    add_in_batches,
    prepare_chromadb_data,
    get_collection_data,
    load_train_test_data,
    delete_collection
)

from .train import (
    train_xgboost,
    train_xgboost_with_cv,
    train_logistic_regression,
    train_mlp,
    save_model,
    load_model
)

from .evaluate import (
    compute_metrics,
    print_metrics,
    get_classification_report,
    get_confusion_matrix,
    compute_overfitting_gap,
    print_overfitting_analysis,
    compute_roc_data,
    plot_roc_curves,
    plot_confusion_matrix,
    evaluate_model
)

__all__ = [
    # Data loading
    'load_airline_sentiment_data',
    'explore_dataframe', 
    'get_label_distribution',
    'encode_labels',
    'apply_label_encoding',
    
    # Augmentation
    'create_synonym_augmenter',
    'augment_text',
    'augment_dataframe',
    'get_class_distribution',
    
    # Embeddings
    'load_embedding_model',
    'generate_embeddings',
    'add_embeddings_to_dataframe',
    'to_2d_array',
    'get_embedding_dimension',
    
    # ChromaDB
    'create_client',
    'get_or_create_collection',
    'add_in_batches',
    'prepare_chromadb_data',
    'get_collection_data',
    'load_train_test_data',
    'delete_collection',
    
    # Training
    'train_xgboost',
    'train_xgboost_with_cv',
    'train_logistic_regression',
    'train_mlp',
    'save_model',
    'load_model',
    
    # Evaluation
    'compute_metrics',
    'print_metrics',
    'get_classification_report',
    'get_confusion_matrix',
    'compute_overfitting_gap',
    'print_overfitting_analysis',
    'compute_roc_data',
    'plot_roc_curves',
    'plot_confusion_matrix',
    'evaluate_model'
]
