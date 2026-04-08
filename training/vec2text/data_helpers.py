#!/usr/bin/env python3
"""Dataset classes for BGE inversion training."""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class BGEEmbeddingsDataset(Dataset):
    """
    Dataset for training hypothesis model.
    
    Loads pre-computed BGE embeddings paired with their source texts.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        texts_path: str,
        tokenizer,
        max_length: int = 128,
    ):
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        
        print(f"Loading texts from {texts_path}...")
        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f]
        
        # Filter out mismatches
        min_len = min(len(self.embeddings), len(self.texts))
        self.embeddings = self.embeddings[:min_len]
        self.texts = self.texts[:min_len]
        
        print(f"Loaded {len(self.texts)} samples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.embeddings[idx]
        
        # Tokenize text for T5 labels
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Replace padding token id with -100 for loss calculation
        labels = tokenized["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_embeddings": torch.tensor(embedding, dtype=torch.float32),
            "labels": labels,
        }


class BGECorrectorDataset(Dataset):
    """
    Dataset for training corrector model.
    
    Loads:
    - Target embeddings (what we want)
    - Target texts (ground truth)
    - Hypotheses (current guesses from hypothesis model)
    - Hypothesis embeddings (re-embedded hypotheses)
    """
    
    def __init__(
        self,
        embeddings_path: str,
        texts_path: str,
        hypotheses_path: str,
        hypothesis_embeddings_path: str,
        tokenizer,
        max_length: int = 128,
    ):
        print("Loading corrector dataset...")
        
        self.target_embeddings = np.load(embeddings_path)
        self.hypothesis_embeddings = np.load(hypothesis_embeddings_path)
        
        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f]
        
        with open(hypotheses_path, "r", encoding="utf-8") as f:
            self.hypotheses = [line.strip() for line in f]
        
        # Ensure all arrays same length
        min_len = min(
            len(self.target_embeddings),
            len(self.hypothesis_embeddings),
            len(self.texts),
            len(self.hypotheses),
        )
        
        self.target_embeddings = self.target_embeddings[:min_len]
        self.hypothesis_embeddings = self.hypothesis_embeddings[:min_len]
        self.texts = self.texts[:min_len]
        self.hypotheses = self.hypotheses[:min_len]
        
        print(f"Loaded {len(self.texts)} samples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Tokenize hypothesis as encoder input
        hyp_tokenized = self.tokenizer(
            self.hypotheses[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Tokenize target as decoder labels
        target_tokenized = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Replace padding with -100 for loss
        labels = target_tokenized["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "target_embedding": torch.tensor(self.target_embeddings[idx], dtype=torch.float32),
            "hypothesis_embedding": torch.tensor(self.hypothesis_embeddings[idx], dtype=torch.float32),
            "input_ids": hyp_tokenized["input_ids"].squeeze(),
            "attention_mask": hyp_tokenized["attention_mask"].squeeze(),
            "labels": labels,
        }
