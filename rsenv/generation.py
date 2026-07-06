"""Batched generation service for concurrent rollouts.

K rollout threads each drive their own live game client at their own pace and
block on GenerationService.generate() when they need the model's next turn. A
single server thread coalesces whatever requests are pending into one
left-padded model.generate call. Decoding is memory-bandwidth bound, so a
batch of K costs roughly the wall-clock of a batch of 1 — this recovers the
~K× that the previous design (every rollout thread serialized behind a
model_lock at batch size 1) left on the table.

Rollouts stay fully asynchronous with respect to the game world: a rollout
that is mid-action simply misses the current batch and joins the next one.
If requests never coincide, throughput degrades to the old serialized flow,
never below it.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class _Request:
    """One rollout's pending generation: input tokens in, generated tokens out."""

    token_ids: list[int]
    ready: threading.Event = field(default_factory=threading.Event)
    new_ids: list[int] | None = None
    error: BaseException | None = None


def trim_at_stop(row: list[int], stop_ids: set[int], pad_id: int) -> list[int]:
    """Cut one batch row's generated tokens at that row's own end.

    In a batched generate, rows that finish early are back-filled with pad_id
    until the longest row finishes. Keep everything through the first stop
    token (inclusive, matching what a batch-1 generate returns); drop the pad
    back-fill.
    """
    kept: list[int] = []
    for tok in row:
        if tok in stop_ids:
            kept.append(tok)
            break
        if tok == pad_id:
            break
        kept.append(tok)
    return kept


class GenerationService:
    """Coalesces concurrent generate() calls into batched model.generate calls.

    All GPU generation happens on the one server thread, so callers never
    contend for the model. Errors raised inside a batch (e.g. CUDA OOM)
    propagate to every caller waiting on that batch.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        coalesce_window_s: float = 0.1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.coalesce_window_s = coalesce_window_s

        stop_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            stop_ids.append(im_end_id)
        self.stop_ids: list[int] = stop_ids
        # Falling back to EOS (HF's own default) is safe for trimming: pad-as-stop
        # rows keep exactly one EOS then cut, same as a batch-1 generate.
        self.pad_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self.generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.pad_id,
            "eos_token_id": stop_ids,
        }

        # None is the stop sentinel
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, name="generation-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def generate(self, token_ids: list[int]) -> list[int]:
        """Block until the service has generated a continuation for token_ids.

        Called from rollout threads. The returned tokens end at the first stop
        token (inclusive) — the same contract as slicing a batch-1
        model.generate output.
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("GenerationService is not running; call start() first")
        request = _Request(token_ids=list(token_ids))
        self._queue.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        assert request.new_ids is not None
        return request.new_ids

    def _serve(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            batch = [request]
            # Coalesce: requests that arrive within the window — or that piled
            # up while the previous batch was generating — ride along.
            deadline = time.monotonic() + self.coalesce_window_s
            stopping = False
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    stopping = True
                    break
                batch.append(nxt)
            self._run_batch(batch)
            if stopping:
                return

    def _run_batch(self, batch: list[_Request]) -> None:
        try:
            results = self._generate_batch([r.token_ids for r in batch])
            for request, new_ids in zip(batch, results):
                request.new_ids = new_ids
        except BaseException as e:
            for request in batch:
                request.error = e
        finally:
            for request in batch:
                request.ready.set()

    def _generate_batch(self, sequences: list[list[int]]) -> list[list[int]]:
        # Left-pad so every row's last real token sits at the same position;
        # the causal model then decodes all rows from an aligned frontier.
        max_len = max(len(seq) for seq in sequences)
        batch_ids = torch.full((len(sequences), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for row, seq in enumerate(sequences):
            batch_ids[row, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, max_len - len(seq) :] = 1

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                batch_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                **self.generate_kwargs,
            )

        stop_set = set(self.stop_ids)
        generated_rows = outputs[:, max_len:].tolist()
        return [trim_at_stop(row, stop_set, self.pad_id) for row in generated_rows]
