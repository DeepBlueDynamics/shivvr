#!/usr/bin/env python3
"""
BGE Corrector Model

T5 model for iterative correction.
Takes (target_embedding, hypothesis_text, hypothesis_embedding) -> corrected_text
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration


class BGECorrectorModel(nn.Module):
    """
    Corrector model: (target_emb, hyp_text, hyp_emb) -> better_text
    
    Architecture:
    1. Project target and hypothesis embeddings
    2. Create prefix tokens from combined embeddings
    3. Concatenate prefix with tokenized hypothesis
    4. T5 encoder processes combined input
    5. T5 decoder generates corrected text
    
    The model learns to "move" the hypothesis toward the target.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        embedding_dim: int = 384,
        num_prefix_tokens: int = 4,
    ):
        super().__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = self.t5.config.d_model
        
        # Project target embedding
        self.target_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        
        # Project hypothesis embedding
        self.hypothesis_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        
        # Combine both embeddings into prefix tokens
        # Input: target_proj + hyp_proj (2 * hidden_size)
        # Output: num_prefix_tokens * hidden_size
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * num_prefix_tokens),
        )
    
    def forward(
        self,
        target_embedding: torch.Tensor,     # (batch, 384) - where we want to be
        hypothesis_embedding: torch.Tensor,  # (batch, 384) - where we are
        input_ids: torch.Tensor,             # (batch, seq) - tokenized hypothesis
        attention_mask: torch.Tensor,        # (batch, seq)
        labels: torch.Tensor = None,         # (batch, seq) - tokenized target
        **kwargs,
    ):
        batch_size = target_embedding.shape[0]
        device = target_embedding.device
        
        # Project embeddings
        target_proj = self.target_projection(target_embedding)      # (batch, hidden)
        hyp_proj = self.hypothesis_projection(hypothesis_embedding)  # (batch, hidden)
        
        # Create prefix from both
        combined = torch.cat([target_proj, hyp_proj], dim=-1)       # (batch, hidden*2)
        prefix = self.prefix_mlp(combined)                          # (batch, hidden*num_prefix)
        prefix = prefix.view(batch_size, self.num_prefix_tokens, self.hidden_size)
        
        # Get hypothesis token embeddings
        hypothesis_embeds = self.t5.encoder.embed_tokens(input_ids)  # (batch, seq, hidden)
        
        # Concatenate: [prefix | hypothesis_tokens]
        encoder_inputs = torch.cat([prefix, hypothesis_embeds], dim=1)
        
        # Extend attention mask for prefix
        prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Encode
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=extended_mask,
            return_dict=True,
        )
        
        # Decode
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(
        self,
        target_embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs,
    ):
        batch_size = target_embedding.shape[0]
        device = target_embedding.device
        
        # Same prefix creation as forward
        target_proj = self.target_projection(target_embedding)
        hyp_proj = self.hypothesis_projection(hypothesis_embedding)
        combined = torch.cat([target_proj, hyp_proj], dim=-1)
        prefix = self.prefix_mlp(combined).view(batch_size, self.num_prefix_tokens, self.hidden_size)
        
        hypothesis_embeds = self.t5.encoder.embed_tokens(input_ids)
        encoder_inputs = torch.cat([prefix, hypothesis_embeds], dim=1)
        
        prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=extended_mask,
            return_dict=True,
        )
        
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            **kwargs,
        )
        
        return outputs

    def save_pretrained(self, path):
        """Save model to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "t5_state_dict": self.t5.state_dict(),
            "target_projection": self.target_projection.state_dict(),
            "hypothesis_projection": self.hypothesis_projection.state_dict(),
            "prefix_mlp": self.prefix_mlp.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "num_prefix_tokens": self.num_prefix_tokens,
                "hidden_size": self.hidden_size,
            }
        }, path)

    def save_hf_compatible(self, output_dir: str) -> None:
        """Save the T5 backbone in HuggingFace format plus BGE metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.t5.save_pretrained(output_dir)
        meta = {
            "model_type": "bge_corrector",
            "embedding_dim": self.embedding_dim,
            "num_prefix_tokens": self.num_prefix_tokens,
            "hidden_size": self.hidden_size,
            "t5_model_name": "t5-base",
        }
        with (output_dir / "bge_corrector_config.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    
    @classmethod
    def from_pretrained(cls, path, model_name: str = "t5-base"):
        """Load model from path."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        model = cls(
            model_name=model_name,
            embedding_dim=config["embedding_dim"],
            num_prefix_tokens=config["num_prefix_tokens"],
        )
        
        model.t5.load_state_dict(checkpoint["t5_state_dict"])
        model.target_projection.load_state_dict(checkpoint["target_projection"])
        model.hypothesis_projection.load_state_dict(checkpoint["hypothesis_projection"])
        model.prefix_mlp.load_state_dict(checkpoint["prefix_mlp"])
        
        return model
