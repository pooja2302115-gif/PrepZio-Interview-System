import torch

from backend.model.model_config import ModelConfig
from backend.model.language_model import PrepZioLanguageModel


def main():
    config = ModelConfig(
        vocab_size=8000,
        max_seq_len=128,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
    )

    model = PrepZioLanguageModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 32),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 32),
    )

    output = model(input_ids, targets)

    print("Model initialized successfully")
    print("Input shape:", input_ids.shape)
    print("Logits shape:", output["logits"].shape)
    print("Loss:", output["loss"].item())

    output["loss"].backward()

    print("Backward pass successful")
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )


if __name__ == "__main__":
    main()