"""Engine loop skeleton."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_serve.engine.batch import BatchKind, BatchPlan
from nano_serve.engine.config import EngineConfig
from nano_serve.engine.request import RequestMetrics, RequestState, RequestStatus
from nano_serve.observability.tracing import nvtx_range
from nano_serve.sampling.base import SamplingParams
from nano_serve.sampling.greedy import GreedySampler
from nano_serve.sampling.topk_topp import TopKTopPSampler
from nano_serve.scheduler.base import ScheduleBudget
from nano_serve.scheduler.chunked_prefill import ChunkedPrefillScheduler
from nano_serve.scheduler.continuous import ContinuousScheduler


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
        self.iteration = 0

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

    def step(
        self,
        *,
        stream_callback: StreamCallback | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> bool:
        if self.config.scheduler == "chunked_prefill":
            return self._step_chunked_prefill(
                stream_callback=stream_callback,
                batch_callback=batch_callback,
            )
        if self.config.scheduler != "continuous":
            raise ValueError("Engine.step currently requires EngineConfig.scheduler='continuous'.")
        if self.config.kv_cache != "none":
            raise ValueError("Phase 4 continuous batching currently requires kv_cache='none'.")
        if not self.waiting and not self.running:
            return False

        scheduler = ContinuousScheduler(self.config.scheduler_policy)
        with self._nvtx("nano_serve.scheduler.continuous"):
            plan = scheduler.schedule(
                waiting=self.waiting,
                running=self.running,
                kv_cache=None,
                budget=ScheduleBudget(
                    max_num_seqs=self.config.max_num_seqs,
                    max_num_batched_tokens=self.config.max_num_batched_tokens,
                ),
            )
        if not plan.request_ids:
            return False

        iteration = self.iteration
        self.iteration += 1
        states = [self._running_state(request_id) for request_id in plan.request_ids]
        now_ns = time.monotonic_ns()
        for state in states:
            if state.metrics.first_scheduled_time_ns is None:
                state.metrics.first_scheduled_time_ns = now_ns
            if state.num_output_tokens == 0 and state.metrics.prefill_start_time_ns is None:
                state.status = RequestStatus.PREFILL
                state.metrics.prefill_start_time_ns = now_ns
            else:
                state.status = RequestStatus.DECODE

        with self._nvtx(_plan_nvtx_name("iteration.continuous", iteration, plan)):
            self._emit_plan_batch(
                batch_callback,
                event="iteration_start",
                iteration=iteration,
                plan=plan,
            )
            runner = self._model_runner()
            with self._nvtx(_plan_nvtx_name("model_forward.continuous", iteration, plan)):
                logits_batch = runner.next_token_logits_batch(
                    plan.input_token_ids,
                    pad_token_id=self._default_pad_token_id(),
                )
        for row, state in enumerate(states):
            if state.num_output_tokens == 0:
                prefill_end_ns = time.monotonic_ns()
                state.metrics.prefill_end_time_ns = prefill_end_ns
                state.metrics.first_token_time_ns = prefill_end_ns
                state.status = RequestStatus.DECODE

            next_token_id = self._sample(logits_batch[row], state.sampling_params)
            state.output_token_ids.append(next_token_id)
            token_time_ns = time.monotonic_ns()
            state.metrics.last_token_time_ns = token_time_ns
            if stream_callback is not None:
                with self._nvtx("nano_serve.stream"):
                    stream_callback(
                        StreamEvent(
                            request_id=state.request_id,
                            token_id=next_token_id,
                            token_index=state.num_output_tokens - 1,
                            timestamp_ns=token_time_ns,
                        )
                    )

            stop_token_ids = self._stop_token_ids(state.sampling_params)
            if next_token_id in stop_token_ids:
                self._finish_continuous_state(state, stop_reason="eos_token")
            elif state.num_output_tokens >= state.max_new_tokens:
                self._finish_continuous_state(state, stop_reason="max_tokens")

        self._emit_plan_batch(
            batch_callback,
            event="iteration_end",
            iteration=iteration,
            plan=plan,
        )
        return True

    def _step_chunked_prefill(
        self,
        *,
        stream_callback: StreamCallback | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> bool:
        if self.config.kv_cache not in {"none", "contiguous"}:
            raise ValueError("chunked prefill currently supports kv_cache='none' or 'contiguous'.")
        if not self.waiting and not self.running:
            return False

        scheduler = ChunkedPrefillScheduler()
        with self._nvtx("nano_serve.scheduler.chunked_prefill"):
            plan = scheduler.schedule(
                waiting=self.waiting,
                running=self.running,
                kv_cache=None,
                budget=ScheduleBudget(
                    max_num_seqs=self.config.max_num_seqs,
                    max_num_batched_tokens=self.config.max_num_batched_tokens,
                    max_prefill_tokens=self.config.max_prefill_chunk_tokens,
                ),
            )
        if not plan.request_ids:
            return False

        iteration = self.iteration
        self.iteration += 1
        states = [self._running_state(request_id) for request_id in plan.request_ids]
        prefill_chunks = _prefill_chunks_by_request(plan)
        now_ns = time.monotonic_ns()
        for state in states:
            if state.metrics.first_scheduled_time_ns is None:
                state.metrics.first_scheduled_time_ns = now_ns
            if state.request_id in prefill_chunks:
                state.status = RequestStatus.PREFILL
                if state.metrics.prefill_start_time_ns is None:
                    state.metrics.prefill_start_time_ns = now_ns
            else:
                state.status = RequestStatus.DECODE

        with self._nvtx(_plan_nvtx_name("iteration.chunked_prefill", iteration, plan)):
            self._emit_plan_batch(
                batch_callback,
                event="iteration_start",
                iteration=iteration,
                plan=plan,
            )
            runner = self._model_runner()
            for state in states:
                if state.is_terminal:
                    continue
                chunk = prefill_chunks.get(state.request_id)
                if chunk is not None:
                    self._run_prefill_chunk(
                        runner,
                        state,
                        chunk,
                        stream_callback=stream_callback,
                    )
                    continue
                self._run_decode_step(
                    runner,
                    state,
                    stream_callback=stream_callback,
                )

            self._emit_plan_batch(
                batch_callback,
                event="iteration_end",
                iteration=iteration,
                plan=plan,
            )
        return True

    def generate_continuous(
        self,
        requests: list[tuple[list[int], SamplingParams | None]],
        *,
        request_ids: list[str] | None = None,
        stream_callback: StreamCallback | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> list[list[int]]:
        if self.config.scheduler != "continuous":
            raise ValueError("EngineConfig.scheduler must be 'continuous' for continuous batching.")
        if request_ids is not None and len(request_ids) != len(requests):
            raise ValueError("request_ids length must match requests length")
        before_finished = len(self.finished)
        for index, (prompt_token_ids, sampling_params) in enumerate(requests):
            self.add_request(
                prompt_token_ids,
                sampling_params,
                request_id=request_ids[index] if request_ids is not None else None,
            )
        while self.step(stream_callback=stream_callback, batch_callback=batch_callback):
            pass
        states = self.finished[before_finished : before_finished + len(requests)]
        if len(states) != len(requests):
            raise RuntimeError("continuous generation finished state count mismatch")
        states_by_id = {state.request_id: state for state in states}
        ordered_states = (
            [states_by_id[request_id] for request_id in request_ids]
            if request_ids is not None
            else states
        )
        return [list(state.output_token_ids) for state in ordered_states]

    def generate_chunked_prefill(
        self,
        requests: list[tuple[list[int], SamplingParams | None]],
        *,
        request_ids: list[str] | None = None,
        stream_callback: StreamCallback | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> list[list[int]]:
        if self.config.scheduler != "chunked_prefill":
            raise ValueError(
                "EngineConfig.scheduler must be 'chunked_prefill' for chunked prefill."
            )
        if request_ids is not None and len(request_ids) != len(requests):
            raise ValueError("request_ids length must match requests length")
        before_finished = len(self.finished)
        for index, (prompt_token_ids, sampling_params) in enumerate(requests):
            self.add_request(
                prompt_token_ids,
                sampling_params,
                request_id=request_ids[index] if request_ids is not None else None,
            )
        while self.step(stream_callback=stream_callback, batch_callback=batch_callback):
            pass
        states = self.finished[before_finished : before_finished + len(requests)]
        if len(states) != len(requests):
            raise RuntimeError("chunked prefill finished state count mismatch")
        states_by_id = {state.request_id: state for state in states}
        ordered_states = (
            [states_by_id[request_id] for request_id in request_ids]
            if request_ids is not None
            else states
        )
        return [list(state.output_token_ids) for state in ordered_states]

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
                with self._nvtx(
                    _request_nvtx_name(
                        "prefill.single",
                        state,
                        num_tokens=len(state.prompt_token_ids),
                    )
                ):
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
                with self._nvtx(
                    _request_nvtx_name(
                        "decode.single",
                        state,
                        token_index=decode_index,
                        num_tokens=len(context_token_ids),
                    )
                ):
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
                with self._nvtx("nano_serve.stream"):
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
        with self._nvtx(
            _batch_nvtx_name(
                "prefill.static_batch",
                iteration=0,
                batch_size=len(states),
                num_tokens=sum(len(context) for context in prefill_contexts),
            )
        ):
            prefill_logits = runner.next_token_logits_batch(
                prefill_contexts,
                pad_token_id=pad_token_id,
            )
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
                with self._nvtx(
                    _batch_nvtx_name(
                        "decode.static_batch",
                        iteration=decode_index,
                        batch_size=len(contexts),
                        num_tokens=sum(len(context) for context in contexts),
                    )
                ):
                    logits = runner.next_token_logits_batch(
                        contexts,
                        pad_token_id=pad_token_id,
                    )
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
                    with self._nvtx("nano_serve.stream"):
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

    def _finish_continuous_state(self, state: RequestState, *, stop_reason: str) -> None:
        state.stop_reason = stop_reason
        if state.metrics.last_token_time_ns is None:
            state.metrics.last_token_time_ns = time.monotonic_ns()
        state.status = RequestStatus.FINISHED
        if state in self.running:
            self.running.remove(state)
        self.finished.append(state)
        runner = self.model_runner
        free = getattr(runner, "free", None)
        if callable(free):
            free(state.request_id)

    def _run_prefill_chunk(
        self,
        runner: Any,
        state: RequestState,
        chunk: dict[str, int],
        *,
        stream_callback: StreamCallback | None,
    ) -> None:
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        final_chunk = chunk_end >= state.num_prompt_tokens
        with self._nvtx(
            _request_nvtx_name(
                "prefill_chunk",
                state,
                num_tokens=chunk_end - chunk_start,
            )
        ):
            prefill_output = runner.prefill_chunk(
                state.prompt_token_ids,
                start=chunk_start,
                end=chunk_end,
                request_id=state.request_id,
                max_decode_tokens=state.max_new_tokens,
                final_chunk=final_chunk,
            )
        prefill_metadata = dict(getattr(prefill_output, "metadata", {}))
        state.block_table = _metadata_block_table(prefill_metadata)
        state.phase_metadata.append(prefill_metadata)
        state.prefill_cursor = chunk_end
        if not final_chunk:
            state.status = RequestStatus.PREFILL
            return

        logits = getattr(prefill_output, "logits", None)
        if logits is None:
            raise RuntimeError("final prefill chunk must return logits")
        state.status = RequestStatus.DECODE
        next_token_id = self._sample(logits[0], state.sampling_params)
        self._append_sampled_token(
            state,
            next_token_id,
            token_index=0,
            stream_callback=stream_callback,
        )

    def _run_decode_step(
        self,
        runner: Any,
        state: RequestState,
        *,
        stream_callback: StreamCallback | None,
    ) -> None:
        if not state.output_token_ids:
            raise RuntimeError("decode step requires at least one generated token")
        decode_index = state.num_output_tokens
        context_token_ids = [*state.prompt_token_ids, *state.output_token_ids]
        with self._nvtx(
            _request_nvtx_name(
                "decode",
                state,
                token_index=decode_index,
                num_tokens=len(context_token_ids),
            )
        ):
            decode_output = runner.decode(
                context_token_ids,
                new_token_id=state.output_token_ids[-1],
                request_id=state.request_id,
            )
        decode_metadata = dict(getattr(decode_output, "metadata", {}))
        state.block_table = _metadata_block_table(decode_metadata)
        state.phase_metadata.append(decode_metadata)
        logits = decode_output.logits
        next_token_id = self._sample(logits[0], state.sampling_params)
        self._append_sampled_token(
            state,
            next_token_id,
            token_index=decode_index,
            stream_callback=stream_callback,
        )

    def _append_sampled_token(
        self,
        state: RequestState,
        token_id: int,
        *,
        token_index: int,
        stream_callback: StreamCallback | None,
    ) -> None:
        state.output_token_ids.append(token_id)
        token_time_ns = time.monotonic_ns()
        if state.metrics.first_token_time_ns is None:
            state.metrics.first_token_time_ns = token_time_ns
            state.metrics.prefill_end_time_ns = token_time_ns
        state.metrics.last_token_time_ns = token_time_ns
        if stream_callback is not None:
            with self._nvtx("nano_serve.stream"):
                stream_callback(
                    StreamEvent(
                        request_id=state.request_id,
                        token_id=token_id,
                        token_index=token_index,
                        timestamp_ns=token_time_ns,
                    )
                )
        stop_token_ids = self._stop_token_ids(state.sampling_params)
        if token_id in stop_token_ids:
            self._finish_continuous_state(state, stop_reason="eos_token")
        elif state.num_output_tokens >= state.max_new_tokens:
            self._finish_continuous_state(state, stop_reason="max_tokens")

    def _running_state(self, request_id: str) -> RequestState:
        for state in self.running:
            if state.request_id == request_id:
                return state
        raise RuntimeError(f"scheduled request is not running: {request_id}")

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

    def _emit_plan_batch(
        self,
        callback: BatchCallback | None,
        *,
        event: str,
        iteration: int,
        plan: BatchPlan,
    ) -> None:
        if callback is None:
            return
        callback(
            BatchEvent(
                event=event,
                iteration=iteration,
                timestamp_ns=time.monotonic_ns(),
                metadata={
                    "batch_kind": plan.kind.value
                    if isinstance(plan.kind, BatchKind)
                    else str(plan.kind),
                    "batch_size": plan.batch_size,
                    "request_ids": plan.request_ids,
                    "num_prefill_tokens": plan.num_prefill_tokens,
                    "num_decode_tokens": plan.num_decode_tokens,
                    "num_running_reqs": len(self.running),
                    "num_waiting_reqs": len(self.waiting),
                    **_batch_plan_padding_metadata(plan),
                    **plan.metadata,
                },
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
        with self._nvtx("nano_serve.sample"):
            if params.top_k is None and params.top_p is None and params.temperature == 1.0:
                return self.greedy_sampler.sample(logits, params)
            return self.topk_topp_sampler.sample(logits, params)

    def _model_runner(self):
        if self.model_runner is None:
            if self.config.model_path is None:
                raise ValueError("EngineConfig.model_path is required for Phase 1 generation.")

            from nano_serve.model.loader import ModelSpec
            from nano_serve.model.torch_runner import TorchModelRunner

            with self._nvtx("nano_serve.model_runner.init"):
                self.model_runner = TorchModelRunner.from_model_spec(
                    ModelSpec(model_path=Path(self.config.model_path), dtype="bfloat16"),
                    kv_cache=self.config.kv_cache,
                    block_size=self.config.block_size,
                )
        return self.model_runner

    def _nvtx(self, name: str):
        return nvtx_range(name, enabled=self.config.benchmark.enable_nvtx)

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


def _plan_nvtx_name(stage: str, iteration: int, plan: BatchPlan) -> str:
    kind = plan.kind.value if isinstance(plan.kind, BatchKind) else str(plan.kind)
    return (
        f"nano_serve.{stage}:iteration={iteration},kind={kind},"
        f"batch={plan.batch_size},prefill_tokens={plan.num_prefill_tokens},"
        f"decode_tokens={plan.num_decode_tokens}"
    )


def _request_nvtx_name(
    stage: str,
    state: RequestState,
    *,
    token_index: int | None = None,
    num_tokens: int | None = None,
) -> str:
    parts = [f"nano_serve.{stage}", f"request={state.request_id}"]
    if token_index is not None:
        parts.append(f"token_index={token_index}")
    if num_tokens is not None:
        parts.append(f"tokens={num_tokens}")
    return ":".join(parts)


def _batch_nvtx_name(
    stage: str,
    *,
    iteration: int,
    batch_size: int,
    num_tokens: int,
) -> str:
    return (
        f"nano_serve.{stage}:iteration={iteration},batch={batch_size},"
        f"tokens={num_tokens}"
    )


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


def _batch_plan_padding_metadata(plan: BatchPlan) -> dict[str, object]:
    lengths = [len(token_ids) for token_ids in plan.input_token_ids]
    max_len = max(lengths, default=0)
    real_tokens = sum(lengths)
    return {
        "max_tokens_per_slot": max_len,
        "real_tokens": real_tokens,
        "padded_tokens": plan.batch_size * max_len - real_tokens,
        "inactive_slots": 0,
    }


def _prefill_chunks_by_request(plan: BatchPlan) -> dict[str, dict[str, int]]:
    raw_chunks = plan.metadata.get("prefill_chunks")
    if not isinstance(raw_chunks, list):
        return {}
    chunks: dict[str, dict[str, int]] = {}
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        request_id = raw_chunk.get("request_id")
        start = raw_chunk.get("start")
        end = raw_chunk.get("end")
        if not isinstance(request_id, str):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        chunks[request_id] = {"start": start, "end": end}
    return chunks
