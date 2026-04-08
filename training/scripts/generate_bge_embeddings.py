#!/usr/bin/env python3
"""Generate GTR-T5-base embeddings for MSMARCO dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from numpy.lib.format import open_memmap
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def mean_pool(hidden_state, attention_mask):
    """Mean pooling with attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # Load dataset
    print("Loading MSMARCO corpus...")
    dataset = load_dataset("mteb/msmarco", "corpus")["corpus"]

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    num_docs = len(dataset)
    print(f"Processing {num_docs} documents...")

    embedding_dim = model.config.hidden_size
    embeddings_path = output_dir / "embeddings.npy"
    embeddings_mmap = open_memmap(
        embeddings_path,
        mode="w+",
        dtype="float32",
        shape=(num_docs, embedding_dim),
    )

    texts_path = output_dir / "texts.txt"
    write_idx = 0
    skipped = 0

    with open(texts_path, "w", encoding="utf-8") as text_file:
        for i in tqdm(range(0, num_docs, args.batch_size)):
            batch = dataset[i:i + args.batch_size]
            texts = batch["text"]

            # Filter empty texts
            cleaned_texts = [t for t in texts if t and len(t.strip()) > 10]
            skipped += len(texts) - len(cleaned_texts)
            if not cleaned_texts:
                continue

            # Tokenize
            inputs = tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            # Embed
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            batch_embeddings = embeddings.cpu().numpy()
            batch_size = batch_embeddings.shape[0]

            embeddings_mmap[write_idx:write_idx + batch_size] = batch_embeddings
            embeddings_mmap.flush()

            for text in cleaned_texts:
                text_file.write(text.replace("\n", " ").replace("\r", " ") + "\n")

            write_idx += batch_size

            if (i // args.batch_size) % 500 == 0:
                print(f"  Processed {i + args.batch_size} documents...")

    metadata = {
        "num_samples": write_idx,
        "num_allocated": num_docs,
        "embedding_dim": embedding_dim,
        "max_length": args.max_length,
        "model_name": args.model_name,
        "dataset": "mteb/msmarco:corpus",
        "skipped_empty": skipped,
        "embeddings_path": str(embeddings_path),
        "texts_path": str(texts_path),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print("Done!")
    print(f"  Embeddings: {write_idx} x {embedding_dim}")
    print(f"  Saved to: {output_dir}")
    if write_idx < num_docs:
        print(f"  Note: {num_docs - write_idx} empty/short documents were skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GTR-T5-base embeddings for MSMARCO")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--output_dir", type=str, default="./embeddings/gtr-t5-base-msmarco")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    
    args = parser.parse_args()
    main(args)
