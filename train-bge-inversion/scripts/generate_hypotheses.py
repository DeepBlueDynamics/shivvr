#!/usr/bin/env python3
"""Generate hypotheses using trained hypothesis model for corrector training."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel


def mean_pool(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def _truncate(text, max_chars):
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _log_samples(batch_texts, batch_hypotheses, args, log_fn):
    if not batch_hypotheses:
        return

    log_fn("Sample hypotheses:")
    count = min(args.sample_count, len(batch_hypotheses))
    for idx in range(count):
        hypothesis = _truncate(batch_hypotheses[idx], args.sample_chars)
        if batch_texts is not None:
            target = _truncate(batch_texts[idx], args.sample_chars)
            log_fn(f"  target: {target}")
            log_fn(f"  hyp:    {hypothesis}")
        else:
            log_fn(f"  hyp: {hypothesis}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load hypothesis model
    print(f"Loading hypothesis model from {args.hypothesis_model_path}...")
    hypothesis_model = BGEInversionModel.from_pretrained(args.hypothesis_model_path).to(device)
    hypothesis_model.eval()
    
    # Tokenizer for T5
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load BGE for re-embedding hypotheses
    print("Loading BGE model...")
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    bge_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device)
    bge_model.eval()
    
    # Load source embeddings
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = np.load(args.embeddings_path)

    if args.max_samples:
        embeddings = embeddings[:args.max_samples]

    print(f"Generating hypotheses for {len(embeddings)} samples...")

    texts = None
    if args.texts_path:
        print(f"Loading texts from {args.texts_path}...")
        with open(args.texts_path, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
        if args.max_samples:
            texts = texts[:args.max_samples]
    
    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_hypotheses = []
    all_hypothesis_embeddings = []
    
    total_samples = len(embeddings)
    total_batches = math.ceil(total_samples / args.batch_size)
    start_time = time.monotonic()

    pbar = tqdm(range(0, total_samples, args.batch_size))
    for batch_idx, i in enumerate(pbar):
        batch_embs = torch.tensor(
            embeddings[i:i + args.batch_size],
            dtype=torch.float32,
        ).to(device)
        
        # Generate hypotheses
        with torch.no_grad():
            generated_ids = hypothesis_model.generate(
                batch_embs,
                max_length=args.max_length,
                num_beams=4,
            )
        
        # Decode to text
        batch_hypotheses = t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_hypotheses.extend(batch_hypotheses)

        batch_texts = None
        if texts is not None:
            batch_texts = texts[i:i + args.batch_size]
        
        # Re-embed hypotheses with BGE
        bge_inputs = bge_tokenizer(
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            bge_outputs = bge_model(**bge_inputs)
            hyp_embs = mean_pool(bge_outputs.last_hidden_state, bge_inputs["attention_mask"])
            hyp_embs = torch.nn.functional.normalize(hyp_embs, p=2, dim=1)
        
        all_hypothesis_embeddings.append(hyp_embs.cpu().numpy())

        if args.log_every and (batch_idx % args.log_every == 0 or batch_idx == total_batches - 1):
            elapsed = time.monotonic() - start_time
            processed = min((batch_idx + 1) * args.batch_size, total_samples)
            avg_len = sum(len(h) for h in batch_hypotheses) / max(1, len(batch_hypotheses))
            pbar.write(
                f"Progress: {processed}/{total_samples} "
                f"({processed / total_samples:.1%}) | "
                f"batch_avg_len={avg_len:.1f} | "
                f"elapsed={elapsed:.0f}s"
            )

        if args.sample_every and (batch_idx % args.sample_every == 0):
            _log_samples(batch_texts, batch_hypotheses, args, pbar.write)
    
    # Concatenate embeddings
    hypothesis_embeddings = np.concatenate(all_hypothesis_embeddings, axis=0)
    
    # Save
    print(f"Saving to {output_dir}...")
    
    with open(output_dir / "hypotheses.txt", "w", encoding="utf-8") as f:
        for h in all_hypotheses:
            f.write(h.replace("\n", " ").replace("\r", " ") + "\n")
    
    np.save(output_dir / "hypothesis_embeddings.npy", hypothesis_embeddings)
    
    print(f"Done!")
    print(f"  Hypotheses: {len(all_hypotheses)}")
    print(f"  Embeddings: {hypothesis_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis_model_path", type=str, required=True, help="Path to trained hypothesis model")
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/bge-small-msmarco/embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="./embeddings/bge-small-msmarco-hypotheses")
    parser.add_argument("--texts_path", type=str, default=None, help="Optional texts file to show targets in logs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100, help="Log status every N batches")
    parser.add_argument("--sample_every", type=int, default=500, help="Show sample outputs every N batches")
    parser.add_argument("--sample_count", type=int, default=2, help="Number of samples to show per log")
    parser.add_argument("--sample_chars", type=int, default=160, help="Max chars to show per sample line")
    
    args = parser.parse_args()
    main(args)
