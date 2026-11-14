import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0


def make_prompt(seed_text: str, n_tokens: int) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": seed_text})
    toks = res.body["tokens"]
    assert len(toks) > 0
    out = []
    while len(out) < n_tokens:
        out.extend(toks)
    return out[:n_tokens]


def complete(prompt: list[int], n_predict: int, timeout: float = DEFAULT_REQUEST_TIMEOUT):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": True,
        "cache_prompt": False,
    }, timeout=timeout)


def wait_for_processing():
    while True:
        res = server.make_request("GET", "/slots")
        if any(s["is_processing"] for s in res.body):
            return


def n_processing() -> int:
    return sum(1 for s in server.make_request("GET", "/slots").body if s["is_processing"])


def test_prompt_waits_for_room_in_the_pool():
    """A prompt that does not fit next to the sequences in flight waits.

    Two prompts that together fill the whole pool (1500 + 1572 = 3072 cells)
    would leave no cell for the next generated token: decode() then halves
    the batch down to 1 and aborts every slot. The second prompt is instead
    launched once the first slot has finished, and both requests succeed.
    The first prompt is still being processed when the second one arrives,
    so the check has to count the sibling's full prompt, not the part of it
    that is already in the cache.
    """
    # the model's training context (2048) caps a slot; the pool is larger
    server.n_ctx = 3072
    server.n_batch = 16
    server.n_ubatch = 16
    server.start()

    prompt_a = make_prompt("The quick brown fox jumps over the lazy dog. ", 1500)
    prompt_b = make_prompt("Once upon a time there was a little girl named Lily. ", 3072 - 1500)

    def second():
        wait_for_processing()
        return complete(prompt_b, 1)

    results = parallel_function_calls([
        (complete, (prompt_a, 200)),
        (second, ()),
    ])
    for res, prompt, n_predict in zip(results, [prompt_a, prompt_b], [200, 1]):
        assert res.status_code == 200
        assert res.body["timings"]["prompt_n"] == len(prompt)
        assert res.body["tokens_predicted"] == n_predict


def test_prompt_that_never_fits_is_rejected_at_once():
    """A prompt larger than a slot is rejected without waiting for the pool.

    Only prompts that could fit an idle pool are held back; one that is
    rejected at slot start anyway (n_tokens >= n_ctx_slot) must not sit in
    the deferred queue until the busy slot has finished.
    """
    server.n_ctx = 512
    server.n_batch = 64
    server.start()

    prompt_a = make_prompt("The quick brown fox jumps over the lazy dog. ", 64)
    prompt_b = make_prompt("Once upon a time there was a little girl named Lily. ", 600)

    def second():
        wait_for_processing()
        # a deferred never-fitting prompt would wait past this timeout
        res = complete(prompt_b, 1, timeout=30)
        # the busy slot is still generating when the rejection arrives
        return res, n_processing()

    results = parallel_function_calls([
        (complete, (prompt_a, 400)),
        (second, ()),
    ])
    assert results[0].status_code == 200
    assert results[0].body["tokens_predicted"] == 400

    res, n_busy = results[1]
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"
    assert n_busy == 1


def test_idle_slot_cells_do_not_block_admission():
    """Cells held by an idle slot are not counted: they are cleared on launch.

    Counting them would hold a prompt that does not fit next to an idle
    conversation forever, since no slot release ever re-checks the deferred
    task.
    """
    server.n_ctx = 512
    server.n_batch = 64
    server.start()

    prompt_a = make_prompt("The quick brown fox jumps over the lazy dog. ", 400)
    prompt_b = make_prompt("Once upon a time there was a little girl named Lily. ", 300)

    res = complete(prompt_a, 1)
    assert res.status_code == 200
    assert n_processing() == 0

    # slot 0 is idle with 401 cells; 401 + 300 does not fit in 512
    res = complete(prompt_b, 1, timeout=30)
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 300
