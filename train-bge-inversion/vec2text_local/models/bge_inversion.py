#!/usr/bin/env python3
"""
BGE Inversion Model

T5 model that takes BGE embeddings as input and generates text.
The embedding is projected and repeated as the encoder input.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration


class BGEInversionModel(nn.Module):
    """
    Hypothesis model: embedding -> text
    
    Architecture:
    1. Project 384-dim BGE embedding to T5 hidden size (768)
    2. Repeat projected embedding num_repeat_tokens times
    3. Use as encoder input to T5
    4. T5 decoder generates text
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        embedding_dim: int = 384,  # bge-small
        num_repeat_tokens: int = 16,
    ):
        super().__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.hidden_size = self.t5.config.d_model
        
        # Project embedding to T5 hidden size
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
    def forward(
        self,
        input_embeddings: torch.Tensor,  # (batch, 384)
        labels: torch.Tensor = None,      # (batch, seq_len)
        **kwargs,
    ):
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Project: (batch, 384) -> (batch, hidden_size)
        projected = self.embedding_projection(input_embeddings)
        
        # Repeat: (batch, hidden_size) -> (batch, num_repeat, hidden_size)
        encoder_inputs = projected.unsqueeze(1).repeat(1, self.num_repeat_tokens, 1)
        
        # Attention mask for encoder (all ones)
        encoder_attention_mask = torch.ones(
            batch_size, self.num_repeat_tokens,
            device=device, dtype=torch.long
        )
        
        # Run T5 encoder
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        
        # Run T5 with encoder outputs
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(
        self,
        input_embeddings: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs,
    ):
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Project and repeat
        projected = self.embedding_projection(input_embeddings)
        encoder_inputs = projected.unsqueeze(1).repeat(1, self.num_repeat_tokens, 1)
        encoder_attention_mask = torch.ones(
            batch_size, self.num_repeat_tokens,
            device=device, dtype=torch.long
        )
        
        # Encode
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        
        # Generate
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
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
            "projection_state_dict": self.embedding_projection.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "num_repeat_tokens": self.num_repeat_tokens,
                "hidden_size": self.hidden_size,
            }
        }, path)

    def save_hf_compatible(self, output_dir: str) -> None:
        """Save the T5 backbone in HuggingFace format plus BGE metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.t5.save_pretrained(output_dir)
        meta = {
            "model_type": "bge_inversion",
            "embedding_dim": self.embedding_dim,
            "num_repeat_tokens": self.num_repeat_tokens,
            "hidden_size": self.hidden_size,
            "t5_model_name": "t5-base",
        }
        with (output_dir / "bge_inversion_config.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    
    @classmethod
    def from_pretrained(cls, path, model_name: str = "t5-base"):
        """Load model from path."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        model = cls(
            model_name=model_name,
            embedding_dim=config["embedding_dim"],
            num_repeat_tokens=config["num_repeat_tokens"],
        )
        
        model.t5.load_state_dict(checkpoint["t5_state_dict"])
        model.embedding_projection.load_state_dict(checkpoint["projection_state_dict"])
        
        return model
