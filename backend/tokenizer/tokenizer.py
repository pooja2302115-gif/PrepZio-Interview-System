from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOKENIZER_PATH = (
    ROOT / "backend" / "data" / "processed" / "tokenizer.json"
)


class BPETokenizer:
    def __init__(self, tokenizer_path=None):
        path = Path(tokenizer_path or DEFAULT_TOKENIZER_PATH)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {path}. "
                "Train it first using backend/tokenizer/train_tokenizer.py"
            )

        self.tokenizer = HFTokenizer.from_file(str(path))

        self.pad_token_id = self._get_id("[PAD]")
        self.unk_token_id = self._get_id("[UNK]")
        self.bos_token_id = self._get_id("[BOS]")
        self.eos_token_id = self._get_id("[EOS]")
        self.user_token_id = self._get_id("[USER]")
        self.assistant_token_id = self._get_id("[ASSISTANT]")
        self.system_token_id = self._get_id("[SYSTEM]")

    def _get_id(self, token):
        token_id = self.tokenizer.token_to_id(token)

        if token_id is None:
            raise ValueError(f"Missing special token: {token}")

        return token_id

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text, add_special_tokens=True):
        result = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
        return result.ids

    def decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def save(self, path):
        self.tokenizer.save(str(path))