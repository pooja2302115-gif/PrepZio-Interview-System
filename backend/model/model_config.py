from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    max_seq_len: int = 512

    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024

    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")