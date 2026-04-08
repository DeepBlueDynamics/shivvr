# Training GTR-T5-Base Inversion Models for shivvr v3

This directory is the clean training workspace for fresh runs.

- Training code lives here and is intended to stay in the repo.
- Generated artifacts such as `.venv/`, `embeddings/`, `models/`, and `saves*/` are ignored and can be recreated.
- Large pre-exported models and old checkpoints were intentionally removed during cleanup.

## Overview

This trains two models:
1. **Hypothesis model**: Takes embedding → generates initial text guess
2. **Corrector model**: Takes (target_embedding, hypothesis, hypothesis_embedding) → generates better text

Both are T5-base models fine-tuned on (text, embedding) pairs from MSMARCO using
`sentence-transformers/gtr-t5-base` embeddings (768d).

---

## Important Notes

- The scripts in `scripts/` are the source of truth; code blocks below are illustrative and can drift.
- Several helper classes in `training/vec2text/` still use legacy `BGE*` names, but the active training stack described here is GTR-T5-base / 768d.
- Default `max_length` is 256. Inversion quality degrades beyond the trained length; for 512-token windows, retrain at 512 or split into overlapping sub-windows.
- ONNX export is two-part: Optimum exports T5 to `t5-onnx/`, and projection/prefix layers are exported separately (`projection.onnx` / `prefix.onnx`). Rust must load both pieces.
- Embedding generation streams to disk (`embeddings.npy`, `texts.txt`) and writes `metadata.json` with the actual sample count. Disk use is ~14GB+ for MSMARCO embeddings; RAM use depends on batch size.

---

## Prerequisites

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download nltk data
python -c "import nltk; nltk.download('punkt')"
```

---

## Hardware Requirements

### Option A: Large GPU (A100 40GB / RTX 3090 24GB)
- Time: ~3-5 days
- Use: `./run_full_training.sh`

### Option B: 12GB GPU (RTX 3080/4080/3060)
- Time: ~5-7 days
- Use: `./run_training_12gb.sh`
- Uses smaller batches + gradient accumulation + mixed precision

### Disk Space
- ~100GB for embeddings + checkpoints
### RAM
- 8-16GB recommended for embedding generation (streamed to disk)

---

## Quick Start

### 1. Verify Setup Works (~10 min)
```bash
chmod +x *.sh
./run_quick_test.sh
```
This runs on 10K samples with 3 epochs. Results will be poor but confirms everything works.

### 2. Full Training

**12GB GPU (most users):**
```bash
./run_training_12gb.sh
```

**24GB+ GPU:**
```bash
./run_full_training.sh
```

### 3. After Training
Models will be in `./saves/`:
- `gtr-t5-base-hypothesis/best_model.pt`
- `gtr-t5-base-corrector/best_model.pt`

---

## Step 1: Generate GTR-T5-Base Embeddings for MSMARCO

First, we need to pre-compute embeddings for the training data.

```python
# scripts/generate_bge_embeddings.py

import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def mean_pool(hidden_state, attention_mask):
    """Mean pooling with attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def main():
    # Config
    model_name = "sentence-transformers/gtr-t5-base"
    dataset_name = "msmarco"
    output_dir = Path("./embeddings/gtr-t5-base-msmarco")
    batch_size = 256
    max_length = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    dataset = load_dataset("mteb/msmarco", "corpus")["corpus"]
    
    # Process in batches
    all_embeddings = []
    all_texts = []
    
    print(f"Generating embeddings for {len(dataset)} documents...")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        texts = batch["text"]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Embed
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # Normalize embeddings before training.
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_texts.extend(texts)
        
        # Save periodically
        if len(all_embeddings) % 100 == 0:
            print(f"Processed {i + batch_size} documents...")
    
    # Concatenate and save
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    print(f"Saving embeddings shape: {embeddings_array.shape}")
    np.save(output_dir / "embeddings.npy", embeddings_array)
    
    # Save texts
    with open(output_dir / "texts.txt", "w") as f:
        for text in all_texts:
            f.write(text.replace("\n", " ") + "\n")
    
    print(f"Done! Saved to {output_dir}")
    print(f"Embeddings: {embeddings_array.shape[0]} x {embeddings_array.shape[1]}")
    print(f"Size: {embeddings_array.nbytes / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python scripts/generate_bge_embeddings.py
```

Outputs (current script):
- `embeddings.npy` (memory-mapped array)
- `texts.txt` (one document per line)
- `metadata.json` (sample count + settings)

---

## Step 2: Create Custom Dataset Loader

```python
# vec2text/data_helpers.py

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class BGEEmbeddingsDataset(Dataset):
    """Dataset that loads pre-computed BGE embeddings."""
    
    def __init__(
        self,
        embeddings_path: str,
        texts_path: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.embeddings = np.load(embeddings_path)
        
        with open(texts_path, "r") as f:
            self.texts = [line.strip() for line in f]
        
        assert len(self.embeddings) == len(self.texts)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.embeddings[idx]
        
        # Tokenize text for T5 output
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_embeddings": torch.tensor(embedding, dtype=torch.float32),
            "labels": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "text": text,
        }


class BGECorrectorDataset(Dataset):
    """Dataset for training corrector with hypotheses."""
    
    def __init__(
        self,
        embeddings_path: str,
        texts_path: str,
        hypotheses_path: str,
        hypothesis_embeddings_path: str,
        tokenizer,
        max_length: int = 256,
    ):
        self.target_embeddings = np.load(embeddings_path)
        self.hypothesis_embeddings = np.load(hypothesis_embeddings_path)
        
        with open(texts_path, "r") as f:
            self.texts = [line.strip() for line in f]
        
        with open(hypotheses_path, "r") as f:
            self.hypotheses = [line.strip() for line in f]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Tokenize hypothesis as input
        hyp_tokenized = self.tokenizer(
            self.hypotheses[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Tokenize target as labels
        target_tokenized = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "target_embedding": torch.tensor(self.target_embeddings[idx], dtype=torch.float32),
            "hypothesis_embedding": torch.tensor(self.hypothesis_embeddings[idx], dtype=torch.float32),
            "input_ids": hyp_tokenized["input_ids"].squeeze(),
            "attention_mask": hyp_tokenized["attention_mask"].squeeze(),
            "labels": target_tokenized["input_ids"].squeeze(),
            "text": self.texts[idx],
            "hypothesis": self.hypotheses[idx],
        }
```

---

## Step 3: Hypothesis Model Architecture

```python
# vec2text/models/bge_inversion.py

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

class BGEInversionModel(nn.Module):
    """
    T5 model that takes GTR-T5-base embeddings as input and generates text.
    
    The embedding is projected and repeated as the encoder input.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        embedding_dim: int = 768,  # gtr-t5-base
        num_repeat_tokens: int = 16,
    ):
        super().__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.num_repeat_tokens = num_repeat_tokens
        
        # Project embedding to T5 hidden size
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.t5.config.d_model),
            nn.GELU(),
            nn.Linear(self.t5.config.d_model, self.t5.config.d_model),
        )
        
    def forward(
        self,
        input_embeddings: torch.Tensor,  # (batch, 768)
        labels: torch.Tensor = None,      # (batch, seq_len)
        attention_mask: torch.Tensor = None,
    ):
        batch_size = input_embeddings.shape[0]
        
        # Project embedding: (batch, 768) -> (batch, d_model)
        projected = self.embedding_projection(input_embeddings)
        
        # Repeat to create encoder inputs: (batch, d_model) -> (batch, num_repeat, d_model)
        encoder_inputs = projected.unsqueeze(1).repeat(1, self.num_repeat_tokens, 1)
        
        # Create attention mask for encoder
        encoder_attention_mask = torch.ones(
            batch_size, self.num_repeat_tokens,
            device=input_embeddings.device
        )
        
        # Run T5 with projected embeddings as encoder input
        outputs = self.t5(
            inputs_embeds=None,
            encoder_outputs=(encoder_inputs,),
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        
        return outputs
    
    def generate(
        self,
        input_embeddings: torch.Tensor,
        max_length: int = 256,
        num_beams: int = 4,
        **kwargs,
    ):
        batch_size = input_embeddings.shape[0]
        
        # Project and repeat
        projected = self.embedding_projection(input_embeddings)
        encoder_inputs = projected.unsqueeze(1).repeat(1, self.num_repeat_tokens, 1)
        encoder_attention_mask = torch.ones(
            batch_size, self.num_repeat_tokens,
            device=input_embeddings.device
        )
        
        # Generate
        outputs = self.t5.generate(
            encoder_outputs=(encoder_inputs,),
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
        
        return outputs

    def save_pretrained(self, path):
        torch.save({
            "t5_state_dict": self.t5.state_dict(),
            "projection_state_dict": self.embedding_projection.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "num_repeat_tokens": self.num_repeat_tokens,
            }
        }, path)
    
    @classmethod
    def from_pretrained(cls, path, model_name="t5-base"):
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        
        model = cls(
            model_name=model_name,
            embedding_dim=config["embedding_dim"],
            num_repeat_tokens=config["num_repeat_tokens"],
        )
        model.t5.load_state_dict(checkpoint["t5_state_dict"])
        model.embedding_projection.load_state_dict(checkpoint["projection_state_dict"])
        
        return model
```

---

## Step 4: Corrector Model Architecture

```python
# vec2text/models/bge_corrector.py

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class BGECorrectorModel(nn.Module):
    """
    T5 model for correction step.
    
    Takes:
    - Target embedding (where we want to be)
    - Hypothesis text (current guess)
    - Hypothesis embedding (where we are)
    
    Outputs corrected text that's closer to target.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        embedding_dim: int = 768,
    ):
        super().__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        # Project both embeddings
        self.target_projection = nn.Linear(embedding_dim, self.t5.config.d_model)
        self.hypothesis_projection = nn.Linear(embedding_dim, self.t5.config.d_model)
        
        # Combine embeddings into prefix
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.t5.config.d_model * 2, self.t5.config.d_model),
            nn.GELU(),
            nn.Linear(self.t5.config.d_model, self.t5.config.d_model * 4),  # 4 prefix tokens
        )
    
    def forward(
        self,
        target_embedding: torch.Tensor,     # (batch, 768)
        hypothesis_embedding: torch.Tensor,  # (batch, 768)
        input_ids: torch.Tensor,             # (batch, seq) - tokenized hypothesis
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        batch_size = target_embedding.shape[0]
        
        # Project embeddings
        target_proj = self.target_projection(target_embedding)  # (batch, d_model)
        hyp_proj = self.hypothesis_projection(hypothesis_embedding)  # (batch, d_model)
        
        # Create prefix from both embeddings
        combined = torch.cat([target_proj, hyp_proj], dim=-1)  # (batch, d_model * 2)
        prefix = self.prefix_mlp(combined)  # (batch, d_model * 4)
        prefix = prefix.view(batch_size, 4, self.t5.config.d_model)  # (batch, 4, d_model)
        
        # Get hypothesis token embeddings
        hypothesis_embeds = self.t5.encoder.embed_tokens(input_ids)  # (batch, seq, d_model)
        
        # Concatenate prefix + hypothesis
        encoder_inputs = torch.cat([prefix, hypothesis_embeds], dim=1)  # (batch, 4+seq, d_model)
        
        # Extend attention mask
        prefix_mask = torch.ones(batch_size, 4, device=attention_mask.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward through T5
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=extended_mask,
        )
        
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask,
            labels=labels,
        )
        
        return outputs
    
    def generate(
        self,
        target_embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 256,
        **kwargs,
    ):
        batch_size = target_embedding.shape[0]
        
        # Same prefix creation as forward
        target_proj = self.target_projection(target_embedding)
        hyp_proj = self.hypothesis_projection(hypothesis_embedding)
        combined = torch.cat([target_proj, hyp_proj], dim=-1)
        prefix = self.prefix_mlp(combined).view(batch_size, 4, self.t5.config.d_model)
        
        hypothesis_embeds = self.t5.encoder.embed_tokens(input_ids)
        encoder_inputs = torch.cat([prefix, hypothesis_embeds], dim=1)
        
        prefix_mask = torch.ones(batch_size, 4, device=attention_mask.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=extended_mask,
        )
        
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask,
            max_length=max_length,
            **kwargs,
        )
        
        return outputs

    def save_pretrained(self, path):
        torch.save({
            "t5_state_dict": self.t5.state_dict(),
            "target_projection": self.target_projection.state_dict(),
            "hypothesis_projection": self.hypothesis_projection.state_dict(),
            "prefix_mlp": self.prefix_mlp.state_dict(),
            "config": {"embedding_dim": self.embedding_dim},
        }, path)
    
    @classmethod
    def from_pretrained(cls, path, model_name="t5-base"):
        checkpoint = torch.load(path)
        model = cls(model_name=model_name, embedding_dim=checkpoint["config"]["embedding_dim"])
        model.t5.load_state_dict(checkpoint["t5_state_dict"])
        model.target_projection.load_state_dict(checkpoint["target_projection"])
        model.hypothesis_projection.load_state_dict(checkpoint["hypothesis_projection"])
        model.prefix_mlp.load_state_dict(checkpoint["prefix_mlp"])
        return model
```

---

## Step 5: Training Script for Hypothesis Model

```python
# scripts/train_hypothesis.py

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.data_helpers_bge import BGEEmbeddingsDataset


def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Wandb
    if args.use_wandb:
        wandb.init(project="shivvr-inversion", name=f"hypothesis-{args.run_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = BGEEmbeddingsDataset(
        embeddings_path=args.embeddings_path,
        texts_path=args.texts_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Split train/val
    train_size = int(0.99 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Model
    print("Creating model...")
    model = BGEInversionModel(
        model_name="t5-base",
        embedding_dim=768,
        num_repeat_tokens=args.num_repeat_tokens,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_embeddings = batch["input_embeddings"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_embeddings=input_embeddings, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_embeddings = batch["input_embeddings"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_embeddings=input_embeddings, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best_model.pt")
            print(f"Saved best model with val_loss={avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    # Save final
    model.save_pretrained(output_dir / "final_model.pt")
    print(f"Training complete! Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/gtr-t5-base-msmarco/embeddings.npy")
    parser.add_argument("--texts_path", type=str, default="./embeddings/gtr-t5-base-msmarco/texts.txt")
    parser.add_argument("--output_dir", type=str, default="./saves/gtr-t5-base-hypothesis")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_repeat_tokens", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="run1")
    
    args = parser.parse_args()
    train(args)
```

---

## Step 6: Generate Hypotheses for Corrector Training

```python
# scripts/generate_hypotheses.py

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel


def mean_pool(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load hypothesis model
    print("Loading hypothesis model...")
    hypothesis_model = BGEInversionModel.from_pretrained(args.hypothesis_model_path).to(device)
    hypothesis_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load embedder for re-embedding hypotheses
    print("Loading GTR-T5-base embedder...")
    bge_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    bge_model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").to(device)
    bge_model.eval()
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = np.load(args.embeddings_path)
    
    with open(args.texts_path, "r") as f:
        texts = [line.strip() for line in f]
    
    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hypotheses = []
    hypothesis_embeddings = []
    
    batch_size = args.batch_size
    
    print(f"Generating hypotheses for {len(embeddings)} samples...")
    
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_embeddings = torch.tensor(
            embeddings[i:i + batch_size],
            dtype=torch.float32
        ).to(device)
        
        # Generate hypotheses
        with torch.no_grad():
            generated_ids = hypothesis_model.generate(
                batch_embeddings,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
        
        # Decode
        batch_hypotheses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        hypotheses.extend(batch_hypotheses)
        
        # Re-embed hypotheses with GTR-T5-base
        bge_inputs = bge_tokenizer(
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            bge_outputs = bge_model(**bge_inputs)
            hyp_embs = mean_pool(bge_outputs.last_hidden_state, bge_inputs["attention_mask"])
            hyp_embs = torch.nn.functional.normalize(hyp_embs, p=2, dim=1)
        
        hypothesis_embeddings.append(hyp_embs.cpu().numpy())
    
    # Save
    hypothesis_embeddings = np.concatenate(hypothesis_embeddings, axis=0)
    
    print(f"Saving hypotheses...")
    with open(output_dir / "hypotheses.txt", "w") as f:
        for h in hypotheses:
            f.write(h.replace("\n", " ") + "\n")
    
    np.save(output_dir / "hypothesis_embeddings.npy", hypothesis_embeddings)
    
    print(f"Done! Saved to {output_dir}")
    print(f"Hypotheses: {len(hypotheses)}")
    print(f"Embeddings shape: {hypothesis_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis_model_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/gtr-t5-base-msmarco/embeddings.npy")
    parser.add_argument("--texts_path", type=str, default="./embeddings/gtr-t5-base-msmarco/texts.txt")
    parser.add_argument("--output_dir", type=str, default="./embeddings/gtr-t5-base-msmarco-hypotheses")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    main(args)
```

---

## Step 7: Training Script for Corrector Model

```python
# scripts/train_corrector.py

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vec2text.models.bge_corrector import BGECorrectorModel
from vec2text.data_helpers_bge import BGECorrectorDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.use_wandb:
        wandb.init(project="shivvr-inversion", name=f"corrector-{args.run_name}")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = BGECorrectorDataset(
        embeddings_path=args.embeddings_path,
        texts_path=args.texts_path,
        hypotheses_path=args.hypotheses_path,
        hypothesis_embeddings_path=args.hypothesis_embeddings_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Split
    train_size = int(0.99 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Model
    print("Creating corrector model...")
    model = BGECorrectorModel(
        model_name="t5-base",
        embedding_dim=768,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            pbar.set_postfix({"loss": loss.item()})
            
            if args.use_wandb:
                wandb.log({"train_loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
            wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best_model.pt")
            print(f"Saved best model")
        
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    model.save_pretrained(output_dir / "final_model.pt")
    print(f"Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="./embeddings/gtr-t5-base-msmarco/embeddings.npy")
    parser.add_argument("--texts_path", type=str, default="./embeddings/gtr-t5-base-msmarco/texts.txt")
    parser.add_argument("--hypotheses_path", type=str, default="./embeddings/gtr-t5-base-msmarco-hypotheses/hypotheses.txt")
    parser.add_argument("--hypothesis_embeddings_path", type=str, default="./embeddings/gtr-t5-base-msmarco-hypotheses/hypothesis_embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="./saves/gtr-t5-base-corrector")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="run1")
    
    args = parser.parse_args()
    train(args)
```

---

## Step 8: Export to ONNX (Full)

This uses Optimum to export the T5 backbone and writes the custom projection/prefix
layers separately as ONNX. Rust should load both pieces.

Prereqs:
```bash
pip install optimum[exporters] onnxruntime
```

Export:
```bash
python scripts/export_onnx_full.py \
  --model_type hypothesis \
  --model_path ./saves/gtr-t5-base-hypothesis/best_model.pt \
  --output_dir ./models/hypothesis

python scripts/export_onnx_full.py \
  --model_type corrector \
  --model_path ./saves/gtr-t5-base-corrector/best_model.pt \
  --output_dir ./models/corrector
```

Outputs:
- `./models/hypothesis/projection.onnx` (embedding -> encoder inputs)
- `./models/hypothesis/t5-onnx/` (Optimum export)
- `./models/hypothesis/export_meta.json` (wiring metadata)
- `./models/corrector/prefix.onnx` (target/hyp embeddings -> prefix)
- `./models/corrector/t5-onnx/`
- `./models/corrector/export_meta.json`

Rust wiring (high level):
- Hypothesis: run `projection.onnx` to get encoder inputs, then run T5 ONNX generation.
- Corrector: run `prefix.onnx` to get prefix tokens, then run T5 ONNX generation with the prefixed hypothesis.
- If your runtime only accepts `input_ids`, you will need a small wrapper to pass `inputs_embeds` to the T5 encoder.

---

## Step 9: Test Inversion

```python
# scripts/test_inversion.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.models.bge_corrector import BGECorrectorModel


def mean_pool(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    print("Loading models...")
    
    bge_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    bge_model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").to(device)
    bge_model.eval()
    
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    hypothesis_model = BGEInversionModel.from_pretrained("./saves/gtr-t5-base-hypothesis/best_model.pt").to(device)
    hypothesis_model.eval()
    
    corrector_model = BGECorrectorModel.from_pretrained("./saves/gtr-t5-base-corrector/best_model.pt").to(device)
    corrector_model.eval()
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Dear Carla, please arrive 30 minutes early for your orthopedic knee surgery.",
        "Uh, so the the vectors are like, you know, 300 to 3000 numbers.",
    ]
    
    for sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"ORIGINAL: {sentence}")
        
        # Embed
        inputs = bge_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = bge_model(**inputs)
            embedding = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Hypothesis
        with torch.no_grad():
            hyp_ids = hypothesis_model.generate(embedding, max_length=256, num_beams=4)
        hypothesis = t5_tokenizer.decode(hyp_ids[0], skip_special_tokens=True)
        print(f"HYPOTHESIS: {hypothesis}")
        
        # Correction loop
        current_text = hypothesis
        num_corrections = 10
        
        for i in range(num_corrections):
            # Embed current hypothesis
            hyp_inputs = bge_tokenizer(current_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                hyp_outputs = bge_model(**hyp_inputs)
                hyp_embedding = mean_pool(hyp_outputs.last_hidden_state, hyp_inputs["attention_mask"])
                hyp_embedding = torch.nn.functional.normalize(hyp_embedding, p=2, dim=1)
            
            # Tokenize hypothesis for corrector input
            t5_inputs = t5_tokenizer(current_text, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device)
            
            # Correct
            with torch.no_grad():
                corrected_ids = corrector_model.generate(
                    target_embedding=embedding,
                    hypothesis_embedding=hyp_embedding,
                    input_ids=t5_inputs["input_ids"],
                    attention_mask=t5_inputs["attention_mask"],
                    max_length=256,
                    num_beams=4,
                )
            current_text = t5_tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
            
            # Compute quality
            final_inputs = bge_tokenizer(current_text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                final_outputs = bge_model(**final_inputs)
                final_embedding = mean_pool(final_outputs.last_hidden_state, final_inputs["attention_mask"])
                final_embedding = torch.nn.functional.normalize(final_embedding, p=2, dim=1)
            
            quality = cosine_similarity(embedding, final_embedding).item()
            print(f"  STEP {i+1}: quality={quality:.4f} | {current_text[:80]}...")
        
        print(f"FINAL: {current_text}")


if __name__ == "__main__":
    main()
```

---

## Full Training Pipeline

```bash
#!/bin/bash
# run_full_training.sh

set -e

echo "=== Step 1: Generate GTR-T5-base embeddings for MSMARCO ==="
python scripts/generate_bge_embeddings.py

echo "=== Step 2: Train hypothesis model ==="
python scripts/train_hypothesis.py \
    --epochs 100 \
    --batch_size 64 \
    --use_wandb

echo "=== Step 3: Generate hypotheses for corrector training ==="
python scripts/generate_hypotheses.py \
    --hypothesis_model_path ./saves/gtr-t5-base-hypothesis/best_model.pt

echo "=== Step 4: Train corrector model ==="
python scripts/train_corrector.py \
    --epochs 100 \
    --batch_size 32 \
    --use_wandb

echo "=== Step 5: Export to ONNX ==="
python scripts/export_onnx_full.py \
    --model_type hypothesis \
    --model_path ./saves/gtr-t5-base-hypothesis/best_model.pt \
    --output_dir ./models/hypothesis

python scripts/export_onnx_full.py \
    --model_type corrector \
    --model_path ./saves/gtr-t5-base-corrector/best_model.pt \
    --output_dir ./models/corrector

echo "=== Step 6: Test ==="
python scripts/test_inversion.py

echo "=== Done! ==="
echo "Models saved to ./models/"
```

---

## Expected Output

After training, you should have:

```
models/
├── hypothesis/
│   ├── projection.onnx
│   ├── t5-onnx/
│   └── export_meta.json
└── corrector/
    ├── prefix.onnx
    ├── t5-onnx/
    └── export_meta.json
```

Upload to HuggingFace:
```bash
huggingface-cli upload shivvr/gtr-t5-base-hypothesis ./models/hypothesis
huggingface-cli upload shivvr/gtr-t5-base-corrector ./models/corrector
```

Then shivvr v3 can pull them in the Dockerfile.
