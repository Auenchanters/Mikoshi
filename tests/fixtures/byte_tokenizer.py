from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from aurora.tokenization.base import TokenizerIdentity


class ByteTokenizer:
    """A deterministic offline tokenizer fixture with a fixed byte mapping."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    @property
    def identity(self) -> TokenizerIdentity:
        settings = {
            "mapping": "special-ids-0-3;utf8-byte-plus-4",
            "special_tokens": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
        }
        payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return TokenizerIdentity(
            kind="test-byte",
            version="1",
            sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            settings=settings,
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [byte + 4 for byte in text.encode("utf-8", errors="strict")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        byte_values: list[int] = []
        for token_id in ids:
            if 0 <= token_id <= 3:
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens cannot be rendered as UTF-8 bytes")
            if not 4 <= token_id < self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the byte fixture vocabulary")
            byte_values.append(token_id - 4)
        return bytes(byte_values).decode("utf-8", errors="strict")
