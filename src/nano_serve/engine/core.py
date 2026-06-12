"""Engine loop skeleton."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nano_serve.engine.config import EngineConfig
from nano_serve.engine.request import RequestMetrics, RequestState, RequestStatus
from nano_serve.sampling.base import SamplingParams
from nano_serve.sampling.greedy import GreedySampler
from nano_serve.sampling.topk_topp import TopKTopPSampler


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    token_id: int
    token_index: int
    timestamp_ns: int


@dataclass(frozen=True)
class PhaseEvent:
    request_id: str
    phase: str
    event: str
    token_index: int | None
    timestamp_ns: int
    num_tokens: int | None = None


StreamCallback = Callable[[StreamEvent], None]
PhaseCallback = Callable[[PhaseEvent], None]


class Engine:
    """Minimal engine shell.

    The real implementation will wire scheduler, KV cache, model runner, and
    sampler according to `EngineConfig`.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.waiting: list[RequestState] = []
        self.running: list[RequestState] = []
        self.finished: list[RequestState] = []
        self.model_runner: Any | None = None
        self.greedy_sampler = GreedySampler()
        self.topk_topp_sampler = TopKTopPSampler()

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or str(uuid.uuid4())
        sampling_params = sampling_params or SamplingParams()
        state = RequestState(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            max_new_tokens=sampling_params.max_tokens,
            metrics=RequestMetrics(arrival_time_ns=time.monotonic_ns()),
        )
        self.waiting.append(state)
        return request_id

    def step(self) -> None:
        raise NotImplementedError("Engine.step is implemented in the scheduler milestones.")

    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        stream_callback: StreamCallback | None = None,
        phase_callback: PhaseCallback | None = None,
    ) -> list[int]:
        """Generate one request with full-context decoding.

        Phase 1 intentionally re-runs the full prompt plus generated tokens for
        every decode step. KV cache and batching are added in later phases.
        """
        request_id = self.add_request(prompt_token_ids, sampling_params)
        state = self.waiting.pop()
        state.status = RequestStatus.PREFILL
        state.metrics.first_scheduled_time_ns = time.monotonic_ns()
        state.metrics.prefill_start_time_ns = state.metrics.first_scheduled_time_ns
        self.running.append(state)

        params = state.sampling_params
        stop_token_ids = set(params.stop_token_ids)
        if not stop_token_ids:
            stop_token_ids = self._default_stop_token_ids()

        runner = self._model_runner()
        generated: list[int] = []
        for decode_index in range(state.max_new_tokens):
            context_token_ids = [*state.prompt_token_ids, *generated]
            if decode_index == 0:
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="prefill",
                    event="start",
                    token_index=None,
                    num_tokens=len(state.prompt_token_ids),
                )
            else:
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="decode",
                    event="start",
                    token_index=decode_index,
                    num_tokens=len(context_token_ids),
                )

            logits = runner.next_token_logits(context_token_ids)
            if decode_index == 0:
                now_ns = time.monotonic_ns()
                state.metrics.prefill_end_time_ns = now_ns
                state.metrics.first_token_time_ns = now_ns
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="prefill",
                    event="end",
                    token_index=None,
                    timestamp_ns=now_ns,
                    num_tokens=len(state.prompt_token_ids),
                )
                state.status = RequestStatus.DECODE
            else:
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="decode",
                    event="end",
                    token_index=decode_index,
                    num_tokens=len(context_token_ids),
                )

            next_token_id = self._sample(logits[0], params)
            generated.append(next_token_id)
            state.output_token_ids.append(next_token_id)
            token_time_ns = time.monotonic_ns()
            state.metrics.last_token_time_ns = token_time_ns
            if stream_callback is not None:
                stream_callback(
                    StreamEvent(
                        request_id=state.request_id,
                        token_id=next_token_id,
                        token_index=decode_index,
                        timestamp_ns=token_time_ns,
                    )
                )
            if next_token_id in stop_token_ids:
                state.stop_reason = "eos_token"
                break

        if state.stop_reason is None:
            state.stop_reason = "max_tokens"
            if state.metrics.last_token_time_ns is None:
                state.metrics.last_token_time_ns = time.monotonic_ns()
        state.status = RequestStatus.FINISHED
        self.running.remove(state)
        self.finished.append(state)

        if state.request_id != request_id:
            raise RuntimeError("Generated request state mismatch.")
        return generated

    def _emit_phase(
        self,
        callback: PhaseCallback | None,
        state: RequestState,
        *,
        phase: str,
        event: str,
        token_index: int | None,
        timestamp_ns: int | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if callback is None:
            return
        callback(
            PhaseEvent(
                request_id=state.request_id,
                phase=phase,
                event=event,
                token_index=token_index,
                timestamp_ns=timestamp_ns or time.monotonic_ns(),
                num_tokens=num_tokens,
            )
        )

    def _sample(self, logits: Any, params: SamplingParams) -> int:
        if params.top_k is None and params.top_p is None and params.temperature == 1.0:
            return self.greedy_sampler.sample(logits, params)
        return self.topk_topp_sampler.sample(logits, params)

    def _model_runner(self):
        if self.model_runner is None:
            if self.config.model_path is None:
                raise ValueError("EngineConfig.model_path is required for Phase 1 generation.")

            from nano_serve.model.loader import ModelSpec
            from nano_serve.model.torch_runner import TorchModelRunner

            self.model_runner = TorchModelRunner.from_model_spec(
                ModelSpec(model_path=Path(self.config.model_path), dtype="bfloat16")
            )
        return self.model_runner

    def _default_stop_token_ids(self) -> set[int]:
        model = getattr(self._model_runner(), "model", None)
        config = getattr(model, "config", None)
        eos_token_id = getattr(config, "eos_token_id", None)
        if eos_token_id is None:
            return set()
        return {int(eos_token_id)}

