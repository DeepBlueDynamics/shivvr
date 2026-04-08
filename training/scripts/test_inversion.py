#!/usr/bin/env python3
"""Test the trained inversion models."""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.models.bge_corrector import BGECorrectorModel


def mean_pool(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load GTR-T5-base embedder
    print("Loading GTR-T5-base model...")
    gtr_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    gtr_model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").to(device)
    gtr_model.eval()
    
    # Load T5 tokenizer
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load hypothesis model
    print(f"Loading hypothesis model from {args.hypothesis_model}...")
    hypothesis_model = BGEInversionModel.from_pretrained(args.hypothesis_model).to(device)
    hypothesis_model.eval()
    
    # Load corrector model
    print(f"Loading corrector model from {args.corrector_model}...")
    corrector_model = BGECorrectorModel.from_pretrained(args.corrector_model).to(device)
    corrector_model.eval()
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Dear Carla, please arrive 30 minutes early for your orthopedic knee surgery on Thursday.",
        "Uh, so the the vectors are like, you know, 300 to 3000 numbers each.",
        "The stock market fell sharply today amid concerns about inflation.",
    ]
    
    print("\n" + "="*80)
    print("TESTING INVERSION")
    print("="*80)
    
    for sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"ORIGINAL: {sentence}")
        
        # Embed with GTR-T5-base
        inputs = gtr_tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        with torch.no_grad():
            outputs = gtr_model(**inputs)
            embedding = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Generate hypothesis
        with torch.no_grad():
            hyp_ids = hypothesis_model.generate(
                embedding,
                max_length=args.max_length,
                num_beams=4,
            )
        
        hypothesis = t5_tokenizer.decode(hyp_ids[0], skip_special_tokens=True)
        
        # Compute initial quality
        hyp_inputs = gtr_tokenizer(
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        with torch.no_grad():
            hyp_outputs = gtr_model(**hyp_inputs)
            hyp_emb = mean_pool(hyp_outputs.last_hidden_state, hyp_inputs["attention_mask"])
            hyp_emb = torch.nn.functional.normalize(hyp_emb, p=2, dim=1)
        
        initial_quality = cosine_similarity(embedding, hyp_emb).item()
        print(f"HYPOTHESIS (q={initial_quality:.4f}): {hypothesis}")
        
        # Correction loop
        current_text = hypothesis
        current_emb = hyp_emb
        
        for step in range(args.num_corrections):
            # Tokenize current hypothesis
            t5_inputs = t5_tokenizer(
                current_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            
            # Correct
            with torch.no_grad():
                corrected_ids = corrector_model.generate(
                    target_embedding=embedding,
                    hypothesis_embedding=current_emb,
                    input_ids=t5_inputs["input_ids"],
                    attention_mask=t5_inputs["attention_mask"],
                    max_length=args.max_length,
                    num_beams=4,
                )
            
            current_text = t5_tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
            
            # Re-embed
            new_inputs = gtr_tokenizer(
                current_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                new_outputs = gtr_model(**new_inputs)
                current_emb = mean_pool(new_outputs.last_hidden_state, new_inputs["attention_mask"])
                current_emb = torch.nn.functional.normalize(current_emb, p=2, dim=1)
            
            quality = cosine_similarity(embedding, current_emb).item()
            
            # Print progress every few steps
            if (step + 1) % 2 == 0 or step == args.num_corrections - 1:
                truncated = current_text[:70] + "..." if len(current_text) > 70 else current_text
                print(f"  STEP {step+1}: q={quality:.4f} | {truncated}")
        
        print(f"FINAL: {current_text}")
        final_quality = cosine_similarity(embedding, current_emb).item()
        print(f"Quality improvement: {initial_quality:.4f} -> {final_quality:.4f} (+{final_quality - initial_quality:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis_model", type=str, default="./saves/gtr-t5-base-hypothesis/best_model.pt")
    parser.add_argument("--corrector_model", type=str, default="./saves/gtr-t5-base-corrector/best_model.pt")
    parser.add_argument("--num_corrections", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    
    args = parser.parse_args()
    main(args)
