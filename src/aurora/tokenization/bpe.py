from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import tokenizers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from aurora.config import TokenizerConfig
from aurora.tokenization.base import TokenizerIdentity

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
EXPECTED_SPECIAL_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}


def _settings(config: TokenizerConfig) -> dict[str, object]:
    return {
        "add_prefix_space": config.add_prefix_space,
        "byte_fallback": "complete-bytelevel-alphabet",
        "case": "preserved",
        "decoder": "byte_level",
        "input_encoding": "utf-8-strict",
        "line_endings": "LF",
        "min_frequency": config.min_frequency,
        "normalization": config.normalization,
        "pre_tokenizer": config.pre_tokenizer,
        "special_tokens": dict(EXPECTED_SPECIAL_IDS),
        "vocab_size": config.vocab_size,
        "whitespace": "preserved",
    }


def _read_corpus(corpus_paths: Sequence[Path]) -> tuple[list[str], str]:
    if not corpus_paths:
        raise ValueError("at least one ordered corpus path is required")
    digest = hashlib.sha256(b"AURORA-BPE-CORPUS-v1\0")
    texts: list[str] = []
    for index, path in enumerate(corpus_paths):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        canonical = text.replace("\r\n", "\n").replace("\r", "\n")
        encoded = canonical.encode("utf-8")
        digest.update(struct.pack(">QQ", index, len(encoded)))
        digest.update(encoded)
        texts.append(canonical)
    if not any(texts):
        raise ValueError("ordered corpus is empty")
    return texts, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_special_ids(tokenizer: Tokenizer) -> None:
    actual = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if actual != EXPECTED_SPECIAL_IDS:
        raise ValueError(f"unexpected BPE special-token IDs: {actual}")


class BPETokenizer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        identity: TokenizerIdentity,
        corpus_manifest_sha256: str,
    ) -> None:
        _verify_special_ids(tokenizer)
        self._tokenizer = tokenizer
        self._identity = identity
        self._corpus_manifest_sha256 = corpus_manifest_sha256

    @property
    def vocab_size(self) -> int:
        return cast(int, self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def corpus_manifest_sha256(self) -> str:
        return self._corpus_manifest_sha256

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = cast(list[int], self._tokenizer.encode(text, add_special_tokens=False).ids)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return cast(
            str,
            self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens),
        )

    @classmethod
    def load(cls, output_dir: Path) -> BPETokenizer:
        tokenizer_path = output_dir / "tokenizer.json"
        metadata_path = output_dir / "metadata.json"
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata_raw, dict):
            raise ValueError("tokenizer metadata must be a JSON object")
        expected_hash = metadata_raw.get("tokenizer_sha256")
        actual_hash = _sha256(tokenizer_path)
        if expected_hash != actual_hash:
            raise ValueError("tokenizer artifact hash does not match metadata")
        settings = metadata_raw.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("tokenizer metadata settings must be a JSON object")
        identity = TokenizerIdentity(
            kind="bpe",
            version=str(metadata_raw.get("tokenizers_version")),
            sha256=actual_hash,
            settings=cast(dict[str, object], settings),
        )
        corpus_hash = metadata_raw.get("corpus_manifest_sha256")
        if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
            raise ValueError("tokenizer metadata has an invalid corpus manifest hash")
        return cls(Tokenizer.from_file(str(tokenizer_path)), identity, corpus_hash)


def train_bpe(
    corpus_paths: Sequence[Path], output_dir: Path, config: TokenizerConfig
) -> BPETokenizer:
    if config.kind != "bpe":
        raise ValueError("train_bpe requires tokenizer.kind='bpe'")
    if config.vocab_size < 260:
        raise ValueError("BPE vocabulary must contain four specials and all 256 bytes")
    if config.normalization != "NFKC" or config.pre_tokenizer != "byte_level":
        raise ValueError("unsupported BPE preprocessing policy")
    if config.add_prefix_space:
        raise ValueError("BPE add_prefix_space must remain false in this phase")

    texts, corpus_hash = _read_corpus(corpus_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    metadata_path = output_dir / "metadata.json"
    if tokenizer_path.exists() or metadata_path.exists():
        raise FileExistsError(f"tokenizer artifacts already exist in {output_dir}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    _verify_special_ids(tokenizer)
    tokenizer.save(str(tokenizer_path), pretty=False)
    tokenizer_hash = _sha256(tokenizer_path)
    settings = _settings(config)
    metadata = {
        "corpus_manifest_sha256": corpus_hash,
        "corpus_order": [str(path) for path in corpus_paths],
        "settings": settings,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizers_version": tokenizers.__version__,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    identity = TokenizerIdentity(
        kind="bpe",
        version=tokenizers.__version__,
        sha256=tokenizer_hash,
        settings=settings,
    )
    return BPETokenizer(tokenizer, identity, corpus_hash)
