"""GenerationService tests: batching, left-pad alignment, trimming, errors.

A stub model stands in for the LLM so the tests exercise exactly the service's
own contract: requests from concurrent threads coalesce into one batched
generate call, each caller gets back its own row trimmed at its own stop
token, and errors inside a batch propagate to every waiting caller.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rsenv.generation import GenerationService, trim_at_stop

EOS = 0
PAD = 9


class StubTokenizer:
    eos_token_id = EOS
    pad_token_id = PAD
    unk_token_id = 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.unk_token_id  # stub vocab has no <|im_end|>


def expected_generation(last_token: int) -> list[int]:
    """What StubModel generates for a row whose last real token is last_token."""
    return [last_token + 100] * (last_token % 3 + 1) + [EOS]


class StubModel:
    """Deterministic echo model mimicking HF batched-generate semantics.

    Each row generates expected_generation(last real token); rows that finish
    before the longest row are back-filled with PAD, exactly as HF generate
    pads early-EOS rows in a batch. Left-padding puts every row's last real
    token at position -1, which the stub asserts — a service that pads on the
    wrong side fails loudly here.
    """

    def __init__(
        self,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.batch_sizes: list[int] = []
        self.entered = entered
        self.release = release
        self.fail = False

    def eval(self) -> "StubModel":
        return self

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        self.batch_sizes.append(input_ids.shape[0])
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait()
        if self.fail:
            raise RuntimeError("stub generation failure")

        rows: list[list[int]] = []
        for r in range(input_ids.shape[0]):
            assert attention_mask is not None and attention_mask[r, -1] == 1, (
                "left-padding must keep the last real token at position -1"
            )
            rows.append(expected_generation(int(input_ids[r, -1])))

        prompt_len = input_ids.shape[1]
        gen_len = max(len(row) for row in rows)
        out = torch.full((input_ids.shape[0], prompt_len + gen_len), PAD, dtype=torch.long)
        out[:, :prompt_len] = input_ids
        for r, tokens in enumerate(rows):
            out[r, prompt_len : prompt_len + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return out


def make_service(model: StubModel, coalesce_window_s: float) -> GenerationService:
    return GenerationService(
        model=model,
        tokenizer=StubTokenizer(),
        max_new_tokens=64,
        temperature=1.0,
        device=torch.device("cpu"),
        coalesce_window_s=coalesce_window_s,
    )


# ─── trim_at_stop ───────────────────────────────────────────────────────────


def test_trim_keeps_through_first_stop_token():
    assert trim_at_stop([5, 6, EOS, 7, 8], {EOS}, PAD) == [5, 6, EOS]


def test_trim_drops_pad_backfill():
    assert trim_at_stop([5, 6, EOS, PAD, PAD], {EOS}, PAD) == [5, 6, EOS]
    assert trim_at_stop([5, 6, PAD, PAD], {EOS}, PAD) == [5, 6]


def test_trim_no_stop_returns_all():
    assert trim_at_stop([5, 6, 7], {EOS}, PAD) == [5, 6, 7]


def test_trim_pad_as_stop_token_is_kept():
    assert trim_at_stop([5, PAD, PAD], {EOS, PAD}, PAD) == [5, PAD]


# ─── GenerationService ──────────────────────────────────────────────────────


def test_single_request_round_trip():
    model = StubModel()
    service = make_service(model, coalesce_window_s=0.01)
    service.start()
    try:
        assert service.generate([3, 4, 5]) == expected_generation(5)
    finally:
        service.stop()


def test_concurrent_requests_coalesce_into_one_batch():
    model = StubModel()
    service = make_service(model, coalesce_window_s=0.5)
    service.start()
    try:
        # Varying lengths *and* varying generation lengths per row: exercises
        # left-padding on input and pad-backfill trimming on output.
        inputs = [[k + 2] * (k + 1) for k in range(8)]  # row k ends in token k+2
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(service.generate, inputs))
    finally:
        service.stop()

    assert results == [expected_generation(k + 2) for k in range(8)]
    assert model.batch_sizes == [8], "all 8 requests should ride in one batch"


def test_requests_pile_up_while_batch_in_flight():
    entered = threading.Event()
    release = threading.Event()
    model = StubModel(entered=entered, release=release)
    service = make_service(model, coalesce_window_s=0.01)
    service.start()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.generate, [5])
            assert entered.wait(timeout=5), "first batch never reached the model"
            # These arrive while batch 1 is generating; they must coalesce
            # into a single second batch, not run one-by-one.
            second = pool.submit(service.generate, [3, 6])
            third = pool.submit(service.generate, [7])
            release.set()
            assert first.result(timeout=5) == expected_generation(5)
            assert second.result(timeout=5) == expected_generation(6)
            assert third.result(timeout=5) == expected_generation(7)
    finally:
        service.stop()

    assert model.batch_sizes == [1, 2]


def test_batch_error_propagates_to_all_callers_and_service_survives():
    model = StubModel()
    model.fail = True
    service = make_service(model, coalesce_window_s=0.2)
    service.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(service.generate, [4]), pool.submit(service.generate, [5])]
            for f in futures:
                with pytest.raises(RuntimeError, match="stub generation failure"):
                    f.result(timeout=5)

        model.fail = False
        assert service.generate([8]) == expected_generation(8)
    finally:
        service.stop()


def test_generate_after_stop_raises():
    service = make_service(StubModel(), coalesce_window_s=0.01)
    service.start()
    service.stop()
    with pytest.raises(RuntimeError, match="not running"):
        service.generate([5])


def test_generate_without_start_raises():
    service = make_service(StubModel(), coalesce_window_s=0.01)
    with pytest.raises(RuntimeError, match="not running"):
        service.generate([5])
