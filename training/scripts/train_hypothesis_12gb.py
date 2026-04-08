#!/usr/bin/env python3
"""
Train hypothesis model - optimized for 12GB GPU.

Changes from full version:
- Smaller batch size (8 instead of 64)
- Gradient accumulation (8 steps = effective batch 64)
- Mixed precision (fp16)
- Gradient checkpointing
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.data_helpers import BGEEmbeddingsDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Mixed precision scaler
    scaler = GradScaler() if args.fp16 else None
    
    if args.use_wandb:
        import wandb
        wandb.init(project="shivvr-inversion", name=f"hypothesis-12gb-{args.run_name}")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Dataset
    print("Loading dataset...")
    full_dataset = BGEEmbeddingsDataset(
        embeddings_path=args.embeddings_path,
        texts_path=args.texts_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Optionally limit dataset size for faster iteration
    if args.max_samples:
        indices = list(range(min(args.max_samples, len(full_dataset))))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Limited to {len(full_dataset)} samples")
    
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps} effective")
    
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
    print("Creating model...")
    model = BGEInversionModel(
        model_name="t5-base",
        embedding_dim=768,
        num_repeat_tokens=args.num_repeat_tokens,
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        model.t5.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Adjust total steps for gradient accumulation
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
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
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_embeddings = batch["input_embeddings"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixed precision forward
            if args.fp16:
                with autocast():
                    outputs = model(input_embeddings=input_embeddings, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_embeddings=input_embeddings, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "train_loss": loss.item() * args.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                    })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_embeddings = batch["input_embeddings"].to(device)
                labels = batch["labels"].to(device)
                
                if args.fp16:
                    with autocast():
                        outputs = model(input_embeddings=input_embeddings, labels=labels)
                else:
                    outputs = model(input_embeddings=input_embeddings, labels=labels)
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Memory stats
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, GPU mem={mem_used:.1f}GB/{mem_reserved:.1f}GB")
        else:
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
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    model.save_pretrained(output_dir / "final_model.pt")
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/gtr-t5-base-msmarco/embeddings.npy")
    parser.add_argument("--texts_path", type=str, default="./embeddings/gtr-t5-base-msmarco/texts.txt")
    parser.add_argument("--output_dir", type=str, default="./saves/gtr-t5-base-hypothesis")
    
    # Batch settings for 12GB GPU
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_repeat_tokens", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    
    # Memory optimizations
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size")
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="run1")
    
    args = parser.parse_args()
    train(args)
