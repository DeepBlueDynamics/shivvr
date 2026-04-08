#!/usr/bin/env python3
"""Train the corrector model."""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_corrector import BGECorrectorModel
from vec2text.data_helpers import BGECorrectorDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.use_wandb:
        import wandb
        wandb.init(project="shivvr-inversion", name=f"corrector-{args.run_name}")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Dataset
    print("Loading dataset...")
    full_dataset = BGECorrectorDataset(
        embeddings_path=args.embeddings_path,
        texts_path=args.texts_path,
        hypotheses_path=args.hypotheses_path,
        hypothesis_embeddings_path=args.hypothesis_embeddings_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Split
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Model
    print("Creating corrector model...")
    model = BGECorrectorModel(
        model_name="t5-base",
        embedding_dim=384,
        num_prefix_tokens=args.num_prefix_tokens,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            target_embedding = batch["target_embedding"].to(device)
            hypothesis_embedding = batch["hypothesis_embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                target_embedding=target_embedding,
                hypothesis_embedding=hypothesis_embedding,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if args.use_wandb:
                import wandb
                wandb.log({"train_loss": loss.item(), "step": global_step})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                target_embedding = batch["target_embedding"].to(device)
                hypothesis_embedding = batch["hypothesis_embedding"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    target_embedding=target_embedding,
                    hypothesis_embedding=hypothesis_embedding,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
        
        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best_model.pt")
            print(f"  -> Saved best model")
        
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    model.save_pretrained(output_dir / "final_model.pt")
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/bge-small-msmarco/embeddings.npy")
    parser.add_argument("--texts_path", type=str, default="./embeddings/bge-small-msmarco/texts.txt")
    parser.add_argument("--hypotheses_path", type=str, default="./embeddings/bge-small-msmarco-hypotheses/hypotheses.txt")
    parser.add_argument("--hypothesis_embeddings_path", type=str, default="./embeddings/bge-small-msmarco-hypotheses/hypothesis_embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="./saves/bge-small-corrector")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_prefix_tokens", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="run1")
    
    args = parser.parse_args()
    train(args)
