from collections import Counter

import pytest
from utils import *

server = ServerPreset.tinyllama2()

N_BATCH = 64


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = N_BATCH
    server.n_ubatch = N_BATCH
    server.n_predict = 1
    server.temperature = 0.0


def make_prompt(seed_text: str, n_tokens: int) -> list[int]:
    # a token-id prompt has an exact length, so the number of batches a slot
    # needs on its own is known: n_tokens / n_batch
    res = server.make_request("POST", "/tokenize", data={"content": seed_text})
    toks = res.body["tokens"]
    assert len(toks) > 0
    out = []
    while len(out) < n_tokens:
        out.extend(toks)
    return out[:n_tokens]


def stream_increments(prompt: list[int], n_predict: int = 1) -> tuple[int, list[int]]:
    # returns (total, per-batch processed increments) for one streamed request.
    # every progress report after the initial one must show progress: a report
    # that repeats the previous value is a batch the slot sat in without a token
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": True,
        "stream": True,
        "return_progress": True,
        "cache_prompt": False,
    })
    total = None
    last = None
    increments = []
    for data in res:
        progress = data.get("prompt_progress", None)
        if progress is None:
            continue
        total = progress["total"]
        if last is None:
            assert progress["processed"] == 0
        else:
            assert progress["processed"] > last, "slot was starved for a batch"
            increments.append(progress["processed"] - last)
        last = progress["processed"]
    assert total is not None
    assert last == total
    return total, increments


def test_concurrent_prompts_share_each_batch():
    """Two prompts arriving together are both advanced in every batch.

    Without a per-slot share, the first slot in iteration order fills the whole
    batch and the second slot's prompt does not start until the first prompt
    has been fully processed: each stream then reports exactly
    n_tokens / n_batch increments of n_batch tokens. With the share, both
    prompts progress in the same batches, so each needs more than
    n_tokens / n_batch batches and its increments are smaller than n_batch.
    """
    server.start()

    n_tokens = 8 * N_BATCH
    prompt_a = make_prompt("The quick brown fox jumps over the lazy dog. ", n_tokens)
    prompt_b = make_prompt("Once upon a time there was a little girl named Lily. ", n_tokens)
    assert prompt_a != prompt_b

    results = parallel_function_calls([
        (stream_increments, (prompt_a,)),
        (stream_increments, (prompt_b,)),
    ])

    for total, increments in results:
        assert total == n_tokens
        assert sum(increments) == n_tokens
        # a slot that owned every batch would need exactly total / n_batch of them
        assert len(increments) > total // N_BATCH
        # and every batch it owned would carry exactly n_batch tokens
        assert min(increments) < N_BATCH

    # the shares are equal: no stream was starved while the other ran
    n_shared_a = sum(1 for x in results[0][1] if x < N_BATCH)
    n_shared_b = sum(1 for x in results[1][1] if x < N_BATCH)
    assert min(n_shared_a, n_shared_b) >= 4


def test_single_prompt_keeps_full_batches():
    """A lone prompt still owns the whole batch (no share is applied)."""
    server.start()

    n_tokens = 4 * N_BATCH
    prompt = make_prompt("The quick brown fox jumps over the lazy dog. ", n_tokens)

    total, increments = stream_increments(prompt)
    assert total == n_tokens
    assert increments == [N_BATCH] * 4


def test_share_excludes_generating_tokens():
    """The share splits the capacity left after the generating slots' tokens.

    With slot A generating, the batch has one token before the prompts are
    added. Splitting n_batch instead of the remaining capacity gives the first
    prompt (n_batch / 2) tokens and the second (n_batch / 2 - 1): the two
    prompts get unequal shares in every batch. Splitting the remaining
    capacity gives both prompts (n_batch - 1) / 2.
    """
    server.n_slots = 3
    server.n_ctx = 3072
    server.server_slots = True
    server.start()

    n_tokens = 8 * N_BATCH
    prompt_a = make_prompt("The quick brown fox jumps over the lazy dog. ", 2 * N_BATCH)
    prompt_b = make_prompt("Once upon a time there was a little girl named Lily. ", n_tokens)
    prompt_c = make_prompt("It was a dark and stormy night in the old castle. ", n_tokens)

    def generate_a():
        # long enough to keep generating while B and C are processed
        return stream_increments(prompt_a, n_predict=400)

    def prompts_bc():
        # A has emitted its first token by the time it appears in /slots as generating;
        # B and C are then started together against the generating slot
        while True:
            res = server.make_request("GET", "/slots")
            if any(s["is_processing"] for s in res.body):
                break
        return parallel_function_calls([
            (stream_increments, (prompt_b,)),
            (stream_increments, (prompt_c,)),
        ])

    results = parallel_function_calls([
        (generate_a, ()),
        (prompts_bc, ()),
    ])

    expected = (N_BATCH - 1) // 2
    for total, increments in results[1]:
        assert total == n_tokens
        # the dominant per-batch increment is the equal share next to A's token
        assert Counter(increments).most_common(1)[0][0] == expected
