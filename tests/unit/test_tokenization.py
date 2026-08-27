from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.config import TokenizerConfig
from aurora.tokenization.bpe import BPETokenizer, train_bpe
from tests.fixtures.byte_tokenizer import ByteTokenizer


def bpe_config(vocab_size: int = 300) -> TokenizerConfig:
    return TokenizerConfig(kind="bpe", vocab_size=vocab_size, min_frequency=2)


@pytest.mark.parametrize("text", ["hello", " café\n", "नमस्ते", "a\t  b", "🚀"])
def test_byte_fixture_round_trips_utf8_without_downloads(text: str) -> None:
    tokenizer = ByteTokenizer()

    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert encoded[0] == 1
    assert encoded[-1] == 2
    assert tokenizer.decode(encoded) == text
    assert tokenizer.identity.kind == "test-byte"


def test_bpe_training_is_byte_identical_for_fixed_order_and_settings(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta\nalpha gamma\nβeta alpha\n", encoding="utf-8")

    first = train_bpe([corpus], tmp_path / "first", bpe_config())
    second = train_bpe([corpus], tmp_path / "second", bpe_config())

    assert first.identity.sha256 == second.identity.sha256
    assert (tmp_path / "first/tokenizer.json").read_bytes() == (
        tmp_path / "second/tokenizer.json"
    ).read_bytes()


def test_bpe_artifact_documents_preprocessing_and_special_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A  café\r\nA café\n", encoding="utf-8", newline="")

    tokenizer = train_bpe([corpus], tmp_path / "tokenizer", bpe_config())

    metadata = json.loads((tmp_path / "tokenizer/metadata.json").read_text(encoding="utf-8"))
    assert metadata["settings"] == {
        "add_prefix_space": False,
        "byte_fallback": "complete-bytelevel-alphabet",
        "case": "preserved",
        "decoder": "byte_level",
        "input_encoding": "utf-8-strict",
        "line_endings": "LF",
        "min_frequency": 2,
        "normalization": "NFKC",
        "pre_tokenizer": "byte_level",
        "special_tokens": {"<bos>": 1, "<eos>": 2, "<pad>": 0, "<unk>": 3},
        "vocab_size": 300,
        "whitespace": "preserved",
    }
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert len(metadata["corpus_manifest_sha256"]) == 64


def test_bpe_round_trips_unseen_valid_utf8(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("plain training words\n", encoding="utf-8")
    tokenizer = train_bpe([corpus], tmp_path / "tokenizer", bpe_config())
    text = "Unseen 🚀 text\nwith  spaces and देवनागरी"

    ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert tokenizer.decode(ids) == text
    loaded = BPETokenizer.load(tmp_path / "tokenizer")
    assert loaded.identity == tokenizer.identity
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_training_rejects_invalid_utf8_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "invalid.txt"
    corpus.write_bytes(b"valid\n\xffinvalid")

    with pytest.raises(UnicodeDecodeError):
        train_bpe([corpus], tmp_path / "tokenizer", bpe_config())
