from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "backend" / "data" / "raw"
OUTPUT_DIR = ROOT / "backend" / "data" / "processed"

TOKENIZER_PATH = OUTPUT_DIR / "tokenizer.json"

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[USER]",
    "[ASSISTANT]",
    "[SYSTEM]",
]


def collect_training_files():
    files = []

    for pattern in ("*.txt", "*.jsonl"):
        files.extend(DATA_DIR.rglob(pattern))

    return sorted(files)


def train_tokenizer():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_training_files()

    if not files:
        raise FileNotFoundError(
            f"No .txt or .jsonl training files found in {DATA_DIR}. "
            "Add Computer Science text data before training the tokenizer."
        )

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(
        files=[str(path) for path in files],
        trainer=trainer,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer training completed.")
    print(f"Files used: {len(files)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {TOKENIZER_PATH}")


if __name__ == "__main__":
    train_tokenizer()