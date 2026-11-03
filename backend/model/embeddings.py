import torch
import torch.nn as nn


class TokenPositionEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embeddings = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, seq_len)

        token_vectors = self.token_embeddings(input_ids)
        position_vectors = self.position_embeddings(positions)

        return self.dropout(token_vectors + position_vectors)