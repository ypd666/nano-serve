"""Structured output reference logits processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class _JSONState(str, Enum):
    START = "start"
    KEY_OR_END = "key_or_end"
    COLON = "colon"
    VALUE = "value"
    COMMA_OR_END = "comma_or_end"
    DONE = "done"
    INVALID = "invalid"


@dataclass(frozen=True)
class JSONGrammarState:
    """Finite-state grammar for flat JSON objects with scalar values."""

    state: _JSONState = _JSONState.START
    depth: int = 0

    @property
    def done(self) -> bool:
        return self.state == _JSONState.DONE

    @property
    def invalid(self) -> bool:
        return self.state == _JSONState.INVALID

    def allowed_token_ids(self, token_ids: dict[str, int]) -> set[int]:
        if self.state == _JSONState.START:
            return {token_ids["{"]}
        if self.state == _JSONState.KEY_OR_END:
            return {token_ids['"'], token_ids["}"]}
        if self.state == _JSONState.COLON:
            return {token_ids[":"]}
        if self.state == _JSONState.VALUE:
            return {
                token_ids['"'],
                token_ids["number"],
                token_ids["true"],
                token_ids["false"],
                token_ids["null"],
            }
        if self.state == _JSONState.COMMA_OR_END:
            return {token_ids[","], token_ids["}"]}
        return set()

    def advance(self, token_id: int, token_ids: dict[str, int]) -> "JSONGrammarState":
        if token_id not in self.allowed_token_ids(token_ids):
            return JSONGrammarState(_JSONState.INVALID, self.depth)
        token_name = _lookup_token_name(token_id, token_ids)
        if self.state == _JSONState.START and token_name == "{":
            return JSONGrammarState(_JSONState.KEY_OR_END, self.depth + 1)
        if self.state == _JSONState.KEY_OR_END:
            if token_name == "}":
                return JSONGrammarState(_JSONState.DONE, self.depth - 1)
            return JSONGrammarState(_JSONState.COLON, self.depth)
        if self.state == _JSONState.COLON:
            return JSONGrammarState(_JSONState.VALUE, self.depth)
        if self.state == _JSONState.VALUE:
            return JSONGrammarState(_JSONState.COMMA_OR_END, self.depth)
        if self.state == _JSONState.COMMA_OR_END:
            if token_name == ",":
                return JSONGrammarState(_JSONState.KEY_OR_END, self.depth)
            return JSONGrammarState(_JSONState.DONE, self.depth - 1)
        return JSONGrammarState(_JSONState.INVALID, self.depth)


class StructuredLogitsProcessor:
    """Mask logits to the token set accepted by a JSON object grammar."""

    def __init__(self, token_ids: dict[str, int] | None = None) -> None:
        self.token_ids = token_ids or default_json_token_ids()
        missing = _REQUIRED_TOKEN_NAMES - set(self.token_ids)
        if missing:
            raise ValueError(f"missing JSON grammar token ids: {sorted(missing)}")

    def allowed_token_ids(self, state: JSONGrammarState) -> set[int]:
        return state.allowed_token_ids(self.token_ids)

    def accepts(self, state: JSONGrammarState, token_id: int) -> bool:
        return token_id in self.allowed_token_ids(state)

    def advance(self, state: JSONGrammarState, token_id: int) -> JSONGrammarState:
        return state.advance(token_id, self.token_ids)

    def mask_logits(self, logits: Any, state: JSONGrammarState) -> Any:
        import torch

        values = torch.as_tensor(logits)
        allowed = self.allowed_token_ids(state)
        masked = torch.full_like(values, float("-inf"))
        valid_ids = [
            token_id
            for token_id in allowed
            if 0 <= token_id < values.shape[-1]
        ]
        if valid_ids:
            index = torch.tensor(valid_ids, device=values.device)
            masked.index_copy_(-1, index, values.index_select(-1, index))
        return masked


_REQUIRED_TOKEN_NAMES = {'"', "{", "}", ":", ",", "number", "true", "false", "null"}


def default_json_token_ids() -> dict[str, int]:
    return {
        "{": 0,
        "}": 1,
        ":": 2,
        ",": 3,
        '"': 4,
        "number": 5,
        "true": 6,
        "false": 7,
        "null": 8,
    }


def _lookup_token_name(token_id: int, token_ids: dict[str, int]) -> str:
    for name, candidate in token_ids.items():
        if candidate == token_id:
            return name
    raise KeyError(f"unknown token id: {token_id}")
