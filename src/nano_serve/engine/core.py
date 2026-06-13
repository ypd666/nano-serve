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
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class BatchEvent:
    event: str
    iteration: int
    timestamp_ns: int
    metadata: dict[str, object]


StreamCallback = Callable[[StreamEvent], None]
PhaseCallback = Callable[[PhaseEvent], None]
BatchCallback = Callable[[BatchEvent], None]


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
            if decode_index == 0:
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="prefill",
                    event="start",
                    token_index=None,
                    num_tokens=len(state.prompt_token_ids),
                )
                prefill_output = runner.prefill(
                    state.prompt_token_ids,
                    request_id=state.request_id,
                    max_decode_tokens=state.max_new_tokens,
                )
                now_ns = time.monotonic_ns()
                state.metrics.prefill_end_time_ns = now_ns
                state.metrics.first_token_time_ns = now_ns
                prefill_metadata = dict(getattr(prefill_output, "metadata", {}))
                state.block_table = _metadata_block_table(prefill_metadata)
                state.phase_metadata.append(prefill_metadata)
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="prefill",
                    event="end",
                    token_index=None,
                    timestamp_ns=now_ns,
                    num_tokens=len(state.prompt_token_ids),
                    metadata=prefill_metadata,
                )
                logits = prefill_output.logits
                state.status = RequestStatus.DECODE
            else:
                context_token_ids = [*state.prompt_token_ids, *generated]
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="decode",
                    event="start",
                    token_index=decode_index,
                    num_tokens=len(context_token_ids),
                )
                decode_output = runner.decode(
                    context_token_ids,
                    new_token_id=generated[-1],
                    request_id=state.request_id,
                )
                decode_metadata = dict(getattr(decode_output, "metadata", {}))
                state.block_table = _metadata_block_table(decode_metadata)
                state.phase_metadata.append(decode_metadata)
                self._emit_phase(
                    phase_callback,
                    state,
                    phase="decode",
                    event="end",
                    token_index=decode_index,
                    num_tokens=len(context_token_ids),
                    metadata=decode_metadata,
                )
                logits = decode_output.logits

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
        free = getattr(runner, "free", None)
        if callable(free):
            free(state.request_id)

        if state.request_id != request_id:
            raise RuntimeError("Generated request state mismatch.")
        return generated

    def generate_static_batch(
        self,
        requests: list[tuple[list[int], SamplingParams | None]],
        *,
        request_ids: list[str] | None = None,
        stream_callback: StreamCallback | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> list[list[int]]:
        if self.config.scheduler != "static_batch":
            raise ValueError("EngineConfig.scheduler must be 'static_batch' for static batching.")
        if self.config.kv_cache != "none":
            raise ValueError("Phase 3 static batching currently requires kv_cache='none'.")
        if not requests:
            return []
        if len(requests) > self.config.max_num_seqs:
            raise ValueError(
                f"static batch size exceeds max_num_seqs: {len(requests)} > {self.config.max_num_seqs}"
            )
        if request_ids is not None and len(request_ids) != len(requests):
            raise ValueError("request_ids length must match requests length")

        states = [
            self._admit_static_request(
                prompt_token_ids,
                sampling_params,
                request_id=request_ids[index] if request_ids is not None else None,
            )
            for index, (prompt_token_ids, sampling_params) in enumerate(requests)
        ]
        if any(state.max_new_tokens <= 0 for state in states):
            raise ValueError("static batch requests must generate at least one token")
        runner = self._model_runner()
        pad_token_id = self._default_pad_token_id()

        self._emit_batch(batch_callback, event="prefill_start", iteration=0, states=states)
        prefill_contexts = [state.prompt_token_ids for state in states]
        prefill_logits = runner.next_token_logits_batch(prefill_contexts, pad_token_id=pad_token_id)
        prefill_end_ns = time.monotonic_ns()
        for state in states:
            state.metrics.prefill_end_time_ns = prefill_end_ns
            state.metrics.first_token_time_ns = prefill_end_ns
            state.status = RequestStatus.DECODE
        self._emit_batch(batch_callback, event="prefill_end", iteration=0, states=states)

        stop_token_ids_by_request = [self._stop_token_ids(state.sampling_params) for state in states]
        max_steps = max(state.max_new_tokens for state in states)
        logits = prefill_logits
        for decode_index in range(max_steps):
            active_indices = [
                index
                for index, state in enumerate(states)
                if not state.is_terminal and decode_index < state.max_new_tokens
            ]
            if not active_indices:
                break
            if decode_index > 0:
                self._emit_batch(
                    batch_callback,
                    event="decode_step_start",
                    iteration=decode_index,
                    states=states,
                )
                contexts = [
                    [*state.prompt_token_ids, *state.output_token_ids]
                    for state in states
                ]
                logits = runner.next_token_logits_batch(contexts, pad_token_id=pad_token_id)
                self._emit_batch(
                    batch_callback,
                    event="decode_step_end",
                    iteration=decode_index,
                    states=states,
                )

            for state_index, state in enumerate(states):
                if state.is_terminal:
                    continue
                if decode_index >= state.max_new_tokens:
                    self._finish_static_state(state, stop_reason="max_tokens")
                    continue
                token_logits = logits[state_index]
                next_token_id = self._sample(token_logits, state.sampling_params)
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
                if next_token_id in stop_token_ids_by_request[state_index]:
                    self._finish_static_state(state, stop_reason="eos_token")
                elif len(state.output_token_ids) >= state.max_new_tokens:
                    self._finish_static_state(state, stop_reason="max_tokens")

        for state in states:
            if not state.is_terminal:
                self._finish_static_state(state, stop_reason="max_tokens")
        self._emit_batch(batch_callback, event="batch_end", iteration=max_steps, states=states)
        self._order_static_finished_suffix(states)
        return [list(state.output_token_ids) for state in states]

    def _admit_static_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None,
        *,
        request_id: str | None = None,
    ) -> RequestState:
        request_id = self.add_request(prompt_token_ids, sampling_params, request_id=request_id)
        state = self.waiting.pop()
        state.status = RequestStatus.PREFILL
        state.metrics.first_scheduled_time_ns = time.monotonic_ns()
        state.metrics.prefill_start_time_ns = state.metrics.first_scheduled_time_ns
        self.running.append(state)
        return state

    def _finish_static_state(self, state: RequestState, *, stop_reason: str) -> None:
        state.stop_reason = stop_reason
        if state.metrics.last_token_time_ns is None:
            state.metrics.last_token_time_ns = time.monotonic_ns()
        state.status = RequestStatus.FINISHED
        if state in self.running:
            self.running.remove(state)
        if state not in self.finished:
            self.finished.append(state)

    def _order_static_finished_suffix(self, states: list[RequestState]) -> None:
        state_ids = {id(state) for state in states}
        self.finished = [state for state in self.finished if id(state) not in state_ids]
        self.finished.extend(states)

    def _emit_batch(
        self,
        callback: BatchCallback | None,
        *,
        event: str,
        iteration: int,
        states: list[RequestState],
    ) -> None:
        if callback is None:
            return
        callback(
            BatchEvent(
                event=event,
                iteration=iteration,
                timestamp_ns=time.monotonic_ns(),
                metadata=_static_batch_metadata(states),
            )
        )

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
        metadata: dict[str, object] | None = None,
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
                metadata=metadata,
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
                ModelSpec(model_path=Path(self.config.model_path), dtype="bfloat16"),
                kv_cache=self.config.kv_cache,
                block_size=self.config.block_size,
            )
        return self.model_runner

    def _default_stop_token_ids(self) -> set[int]:
        model = getattr(self._model_runner(), "model", None)
        config = getattr(model, "config", None)
        eos_token_id = getattr(config, "eos_token_id", None)
        if eos_token_id is None:
            return set()
        return {int(eos_token_id)}

    def _stop_token_ids(self, params: SamplingParams) -> set[int]:
        stop_token_ids = set(params.stop_token_ids)
        if not stop_token_ids:
            stop_token_ids = self._default_stop_token_ids()
        return stop_token_ids

    def _default_pad_token_id(self) -> int:
        model = getattr(self._model_runner(), "model", None)
        config = getattr(model, "config", None)
        pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_ids = self._default_stop_token_ids()
        return next(iter(eos_token_ids), 0)


def _metadata_block_table(metadata: dict[str, object]) -> list[int]:
    blocks = metadata.get("kv_blocks_used")
    if not isinstance(blocks, int) or blocks <= 0:
        return []
    return list(range(blocks))


def _static_batch_metadata(states: list[RequestState]) -> dict[str, object]:
    lengths = [len(state.prompt_token_ids) + len(state.output_token_ids) for state in states]
    max_len = max(lengths, default=0)
    real_tokens = sum(lengths)
    batch_size = len(states)
    active_slots = sum(0 if state.is_terminal else 1 for state in states)
    return {
        "batch_size": batch_size,
        "active_slots": active_slots,
        "inactive_slots": batch_size - active_slots,
        "max_tokens_per_slot": max_len,
        "real_tokens": real_tokens,
        "padded_tokens": batch_size * max_len - real_tokens,
    }

