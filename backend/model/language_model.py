import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.model.model_config import ModelConfig
from backend.model.embeddings import TokenPositionEmbeddings
from backend.model.transformer import TransformerBlock


class PrepZioLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embeddings = TokenPositionEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        self.output_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Share input token embeddings and output projection weights.
        self.output_head.weight = (
            self.embeddings.token_embeddings.weight
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Input exceeds maximum sequence length")

        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids"
                )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
        }