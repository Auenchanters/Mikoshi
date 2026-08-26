from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TokenizerIdentity:
    kind: str
    version: str
    sha256: str
    settings: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "version": self.version,
            "sha256": self.sha256,
            "settings": dict(self.settings),
        }


class TokenizerProtocol(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def identity(self) -> TokenizerIdentity: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...
