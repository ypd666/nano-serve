"""In-process RPC reference for single-node distributed tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RPCCallRecord:
    method: str
    args_count: int
    kwargs_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "args_count": self.args_count,
            "kwargs_keys": list(self.kwargs_keys),
        }


class LocalRPCServer:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}
        self.calls: list[RPCCallRecord] = []

    def register(self, method: str, handler: Callable[..., Any]) -> None:
        if method in self._handlers:
            raise ValueError(f"duplicate RPC method: {method}")
        self._handlers[method] = handler

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            handler = self._handlers[method]
        except KeyError as exc:
            raise KeyError(f"unknown RPC method: {method}") from exc
        self.calls.append(
            RPCCallRecord(
                method=method,
                args_count=len(args),
                kwargs_keys=tuple(sorted(kwargs)),
            )
        )
        return handler(*args, **kwargs)

    def serve(self) -> None:
        return None


class RPCClient:
    def __init__(self, server: LocalRPCServer | None = None) -> None:
        self.server = server or LocalRPCServer()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.server.call(method, *args, **kwargs)


RPCServer = LocalRPCServer

