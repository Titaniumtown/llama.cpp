import collections
import os
import re
import threading
import time

import pytest
from utils import *

# the adaptive step scheduler (patches/llamacpp/0244) is off unless its env vars are
# set, and these tests pin the *mechanism*: which batch composition each knob picks.
# They do not pin an economics verdict -- the ride-vs-hold criterion is measured per
# machine and model (on the tiny test model a decode step is nearly free, so the
# estimate there is expected to prefer holding, which the production box does not).
N_BATCH = 64
N_PROMPT = 8 * N_BATCH
N_PREDICT = 400

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server(monkeypatch):
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = N_BATCH
    server.n_ubatch = N_BATCH
    server.n_predict = 1
    server.temperature = 0.0

    # the scheduler reads its configuration once, when the server process starts
    for key in list(os.environ):
        if key.startswith("LLAMA_SERVER_STEP_SCHED_"):
            monkeypatch.delenv(key)
    yield monkeypatch


def arm(monkeypatch, **kw):
    for name, value in kw.items():
        monkeypatch.setenv("LLAMA_SERVER_STEP_SCHED_" + name, str(value))


def make_prompt(seed_text: str, n_tokens: int) -> list[int]:
    # a token-id prompt has an exact length, so the number of batches a slot needs is
    # known: n_tokens / n_batch
    res = server.make_request("POST", "/tokenize", data={"content": seed_text})
    toks = res.body["tokens"]
    assert len(toks) > 0
    out = []
    while len(out) < n_tokens:
        out.extend(toks)
    return out[:n_tokens]


def stream_decode(prompt: list[int], n_predict: int) -> list[float]:
    """A streaming completion with per-token timestamps (the decoder under test)."""
    seen = []
    for data in server.make_stream_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "ignore_eos": True,
        "cache_prompt": False,
        "temperature": 0.0,
        "seed": 42,
    }):
        if data.get("content"):
            seen.append(time.time())
    return seen


def stream_prompt(prompt: list[int], delay: float = 0.05, cache: bool = False) -> list[tuple]:
    """A prompt-carrying request, started while the decoder is already generating.

    Returns (timestamp, processed) for every progress report: a report whose
    `processed` repeats the previous value is an iteration the slot sat out (the
    scheduler's injected decode-only steps).
    """
    time.sleep(delay)
    prog = []
    for data in server.make_stream_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 1,
        "stream": True,
        "return_progress": True,
        "ignore_eos": True,
        "cache_prompt": cache,
        "temperature": 0.0,
        "seed": 42,
    }):
        p = data.get("prompt_progress")
        if p is not None:
            prog.append((time.time(), p["processed"]))
    return prog


def measure_rate(server) -> float:
    # decode tokens/s on this machine, measured live: the admission constants are priced in
    # tokens through it, because the box's single-core speed varies ~60x under load and no
    # fixed millisecond floor keeps a one-step arrival and a hundred-step one on opposite
    # sides of the bypass bound
    p = make_prompt("The rate probe for the scheduler. ", 32)
    t0 = time.time()
    stream_decode(p, 256)
    dt = time.time() - t0
    assert dt > 0.05, f"the rate probe was too short to trust: {dt:.3f}s"
    return 256.0 / dt


def measure_pp_rate(server) -> float:
    # prompt tokens/s on this machine, measured live with no decoder in the batch: the
    # incumbent's length and the arrival delays are priced in it, because a fixed-millisecond
    # test dies on a fast box (the incumbent finishes before the newcomer arrives) and
    # hangs on a loaded one
    p = make_prompt("The pp-rate probe for the scheduler. ", 256)
    t0 = time.time()
    r = stream_prompt(p, 0.0)
    dt = r[-1][0] - t0
    assert r[-1][1] == len(p) and dt > 0.05, f"the pp-rate probe was not trustworthy: {dt:.3f}s"
    return float(r[-1][1]) / dt


def measure_shared_pp_rate(server) -> float:
    # prompt tokens/s while a decoder shares the single-threaded batch: the C++ bypass
    # estimate is priced in shared prompt steps (t_chunk is the EWMA of steps that ride
    # with a decoder), so the bypass bound must be measured the same way -- a solo decode
    # probe prices it ~10x too small on a box where sharing is the common case
    dec_p = make_prompt("The shared-rate probe decoder. ", 32)
    pp_p = make_prompt("The shared-rate probe prompt. ", 256)
    got = {}

    def probe():
        t0 = time.time()
        r = stream_prompt(pp_p, 0.0)
        got["rate"] = float(r[-1][1]) / (r[-1][0] - t0)

    dec = threading.Thread(target=stream_decode, args=(dec_p, 512))
    dec.start()
    time.sleep(0.15)
    pp = threading.Thread(target=probe)
    pp.start()
    pp.join()
    dec.join()
    assert got.get("rate"), "the shared prompt probe produced no rate"
    return got["rate"]


def run_overlap() -> dict:
    """Run a long prompt against an in-flight decoder and measure both sides."""
    prompt_dec = make_prompt("The quick brown fox jumps over the lazy dog. ", 32)
    prompt_pre = make_prompt("Once upon a time there was a little girl named Lily. ", N_PROMPT)

    dec_times, prog = parallel_function_calls([
        (stream_decode, (prompt_dec, N_PREDICT)),
        (stream_prompt, (prompt_pre,)),
    ])

    assert len(prog) >= 3, "the prompt produced no progress reports"
    increments = [b[1] - a[1] for a, b in zip(prog, prog[1:])]
    t1, t2 = prog[0][0], prog[-1][0]

    # a decoder gap counts as "during" when it overlaps the prompt window at all: a
    # decoder that stalls for the whole prompt has exactly one such gap
    pairs = [(a, b) for a, b in zip(dec_times, dec_times[1:])]
    during = sorted(b - a for a, b in pairs if a < t2 and b > t1)

    return {
        "increments": increments,
        "injected": sum(1 for x in increments if x == 0),
        "incr_max": max(increments),
        "prompt_tokens": prog[-1][1],
        # the stream coalesces chunks, so the token count is a lower bound on what
        # the decoder emitted; every assertion below allows that slack
        "decode_chunks": len(dec_times),
        "gap_during_max": during[-1] if during else 0.0,
    }


def test_default_composition_is_unchanged():
    """With no scheduler env, a decoder keeps its batch slot and nothing is injected."""
    server.start()
    r = run_overlap()

    assert r["prompt_tokens"] == N_PROMPT
    assert r["injected"] == 0
    # the decoder's pending token occupies one slot of every prompt batch
    assert r["incr_max"] == N_BATCH - 1
    assert r["decode_chunks"] >= N_PREDICT - 5, r


def test_itl_target_injects_decode_only_steps(monkeypatch):
    """A decoder that would wait a whole chunk gets steps of its own in between."""
    arm(monkeypatch, ITL_MS=1, MAX_SHARE=50)
    server.start()
    r = run_overlap()

    assert r["injected"] >= 1, f"no decode-only step was injected: {r}"
    # injections do not starve the prompt: it still completes in its own batches
    assert r["prompt_tokens"] == N_PROMPT
    assert r["decode_chunks"] >= N_PREDICT - 5, r
    # and the decoder really was served in between: it advances more often per unit of
    # prompt progress than the ride-along composition allows
    assert r["injected"] + r["prompt_tokens"] // N_BATCH <= len(r["increments"]) + 1


def test_max_share_zero_never_injects(monkeypatch):
    """The injected share is capped: at 0% the prompt owns every iteration."""
    arm(monkeypatch, ITL_MS=1, MAX_SHARE=0)
    server.start()
    r = run_overlap()

    assert r["injected"] == 0, r
    assert r["incr_max"] == N_BATCH - 1, r
    assert r["prompt_tokens"] == N_PROMPT, r


def test_hold_gives_the_prompt_the_whole_batch(monkeypatch):
    """A held decoder contributes nothing, so the prompt's batches are full."""
    arm(monkeypatch, HOLD=2)
    server.start()
    r = run_overlap()

    assert r["incr_max"] == N_BATCH, "the prompt did not get the decoder's batch slot"
    assert r["prompt_tokens"] == N_PROMPT
    # the held decoder waits for the prompt: one gap spans most of the prompt window
    assert r["gap_during_max"] > 0.0
    assert r["decode_chunks"] >= N_PREDICT - 5, r


def test_hold_estimate_probes_and_releases(monkeypatch):
    """The estimate-gated hold identifies the counterfactual instead of guessing.

    A decoder that always rides leaves no "prompt step without a decoder" sample, so
    the scheduler has to take held probes to measure one. Both compositions must show
    up in the progress stream, and the decoder must resume afterwards.
    """
    arm(monkeypatch, HOLD=1, PROBE_EVERY=2)
    server.start()
    r = run_overlap()

    assert r["prompt_tokens"] == N_PROMPT
    assert r["decode_chunks"] >= N_PREDICT - 5, r
    # a probe shows up as a full batch (the decoder was held out of it)
    assert r["incr_max"] == N_BATCH, "no held probe was taken"
    # and the decoder kept riding in between, so the prompt is not starved either
    assert any(x < N_BATCH for x in r["increments"]), "the scheduler held every iteration"


def test_admission_floor_holds_while_the_decoders_spend_the_target(monkeypatch, tmp_path):
    """The floor is the budget seen from the other side: while the decoders in the batch already
    spend more of the target than the floor allows, a prompt that has not started waits -- and it
    still finishes.

    The target is set to 1 ms because the toy model cannot spend the production 400 ms one: its
    whole decode step costs ~2.4 ms against a 400 ms target, i.e. every arrival would be under
    the floor and nothing would ever be held. At 1 ms the decoders overshoot by an order of
    magnitude, the hold is deterministic, and ADMIT_BYPASS_MS=0 keeps the short-prefill bypass
    out of the way.
    """
    log = tmp_path / "sched-admit-live.log"
    arm(monkeypatch, ITL_MS=1, MAX_SHARE=33, ADMIT_TG_PCT=1, ADMIT_PP_PCT=0,
        ADMIT_WAIT_MS=400, ADMIT_BYPASS_MS=0, LOG=64)
    server.n_threads = 1
    server.n_ctx = 8192
    server.log_path = str(log)
    server.start()

    dec_prompt = make_prompt("The quick brown fox jumps over the lazy dog. ", 32)
    subject = make_prompt("It was a dark and stormy night in the old castle. ", 256)

    dec_thread = threading.Thread(target=stream_decode, args=(dec_prompt, 600))
    dec_thread.start()
    time.sleep(0.10)

    t0 = time.time()
    r = stream_prompt(subject, 0.0)
    subj_wait = r[0][0] - t0
    dec_thread.join()
    server.stop()

    assert r[-1][1] == len(subject), f"the held prompt never finished: {r[-1][1]}"
    text = log.read_text()
    ev = [l.split("]: ", 1)[-1].strip()[:150] for l in text.splitlines() if "admission" in l]
    assert "admission hold" in text, \
        "no arrival was held while the decoders spent the target:\n" + "\n".join(ev)
    m = re.search(r"decoders spend (\d+)% of the target", text)
    assert m, "the hold line did not report the decoders' share of the target:\n" + "\n".join(ev)
    assert int(m.group(1)) >= 1, f"the share was under the floor and the arrival was still held: {m.group(1)}%"
    assert subj_wait >= 0.10, f"the held arrival did not wait at all: {subj_wait:.2f}s\n" + "\n".join(ev)


def test_admission_pp_floor_holds_second_prompt(monkeypatch, tmp_path):
    """A prompt that arrives while another is processing waits, bounded like the tg floor.

    This pins the hold path with a PROMPT incumbent and no decoder in the batch (the tg
    test covers the decoder case). The measurement-driven behaviour on this test model is
    asserted by test_state_dump_is_emitted instead: the measured sharing ratio comes out at
    parity here, because the per-step overhead dominates at these sizes -- exactly what the
    production box measures too (0.95), where a 50% floor correctly never fires.

    The incumbent fills the per-slot context (1024 tokens with two slots), so its prefill duration
    spans 0.25 s on a fast box to 2 minutes on a loaded one; every other time is a fraction of it
    of that duration, so the hold invariants hold at any single-core speed.
    """
    log = tmp_path / "sched-admit-pp.log"
    server.n_ctx = 2048
    server.n_slots = 2
    # one thread pins the step rate on any machine
    server.n_threads = 1
    server.n_batch = 2
    server.n_ubatch = 2
    server.log_path = str(log)
    server.start()
    rate = measure_pp_rate(server)
    server.stop()

    def pms(n_tokens: int) -> int:
        return int(n_tokens * 1000.0 / rate)

    # the bound must elapse while the incumbent is still in flight: its remaining work at the
    # subject's arrival is half the incumbent's duration and the bound is two-fifths, so the
    # wait is the bound, deterministic at any speed
    arm(monkeypatch, ADMIT_TG_PCT=0, ADMIT_PP_PCT=100, ADMIT_FORCE=1,
        ADMIT_WAIT_MS=pms(400), LOG=64)
    server.start()

    long_prompt = make_prompt("Once upon a time there was a little girl named Lily. ", 1000)
    short_prompt = make_prompt("The quick brown fox jumps over the lazy dog. ", 128)
    subject = make_prompt("It was a dark and stormy night in the old castle. ", 128)

    def job(prompt):
        def run():
            start = time.time()
            r = stream_prompt(prompt, 0.0)
            return start, (r[0][0] if r else None), (r[-1][1] if r else None)
        return run

    # `ADMIT_FORCE` holds every newcomer while anything is in flight, so the shape is
    # deterministic: the first prompt finds an empty box and starts at once, the two that
    # arrive while it is still processing wait out the bound. the delays are priced in
    # tokens through the live rate probe: 250 is a quarter of the incumbent's 1000, 500 half

    def delayed(fn, n_tokens):
        def run():
            time.sleep(n_tokens / rate)
            return fn()
        return run

    results = parallel_function_calls([
        (job(long_prompt), ()),
        (delayed(job(short_prompt), 250), ()),
        (delayed(job(subject), 500), ()),
    ])
    first_start, first_first, first_done = results[0]
    short_start, short_first, short_done = results[1]
    subj_start, subj_first, subj_done = results[2]

    assert first_first - first_start < pms(64) / 1000, "the first prompt found an empty box and should not have waited"
    assert first_done == len(long_prompt), f"the incumbent never finished: {first_done}"
    assert short_done == len(short_prompt) and subj_done == len(subject), "a held prompt never finished"

    assert subj_first - subj_start >= pms(200) / 1000, f"the newcomer was not held: waited {subj_first - subj_start:.2f}s"


def test_admission_bypasses_a_short_cache_hit(monkeypatch, tmp_path):
    """Decode in one slot, a cache-hit follow-up in the other: take the hit, don't wait.

    The floors protect an incumbent from *sustained* competition, and what a newcomer costs is
    proportional to its own remaining prompt work. A follow-up on a cached conversation is a
    few hundred tokens (a question, a tool call), i.e. one prompt step; holding it for the
    decoder's remaining minutes would cost far more than the one chunk-length hiccup it causes.

    The two probes here differ only in their cache state: identical tokens, and the cold one
    has to pay for a whole prefill while the follow-up is one step.

    The C++ estimate prices a newcomer in shared prompt steps (a step that rides with the
    decoder), so the bypass and wait bounds are priced in the same unit -- prompt tokens/s
    measured against a live decoder -- not in solo decode tokens, which on a single thread
    prices the bound ~10x below the estimate and holds the one-step arrival.
    """
    log = tmp_path / "sched-bypass.log"
    # small batches so the two estimates separate by ~125x (a cache hit is one step, a cold
    # 1000-token prompt is 125 of them); the admission constants are priced in shared prompt
    # tokens through a live probe, because the box's single-core speed varies ~60x under load
    # and no fixed millisecond floor keeps the two on opposite sides of the bypass bound
    server.n_batch = 8
    server.n_ubatch = 8
    server.n_ctx = 16384
    # one thread pins the rate; the 1600-token decoder stays inside the model's 2048-token
    # context, so no ctx shift is needed to keep it alive past the probes
    server.n_threads = 1
    server.log_path = str(log)
    server.start()
    rate = measure_shared_pp_rate(server)
    server.stop()

    def pms(n_tokens: int) -> int:
        return int(n_tokens * 1000.0 / rate)

    arm(monkeypatch, ITL_MS=1, ADMIT_TG_PCT=1, ADMIT_PP_PCT=0,
        ADMIT_WAIT_MS=pms(256), ADMIT_BYPASS_MS=pms(64), LOG=64)
    server.start()

    base = make_prompt("Once upon a time there was a little girl named Lily. ", 1000)
    follow = base + make_prompt("It was a dark and stormy night in the old castle. ", 8)
    decoder = make_prompt("It was a dark and stormy night in the old castle. ", 32)

    # the decoder takes one slot and stays for seconds; every probe below runs on the other one
    dec_thread = threading.Thread(target=stream_decode, args=(decoder, 1600))
    dec_thread.start()
    time.sleep(0.15)

    # 1. a cold prompt, run to leave its tokens cached in the slot; probe 1's own wait is not
    #    asserted: the live floor is deterministic, so a cold prompt waits like any other
    r = stream_prompt(base, 0.0, cache=True)

    # 2. the same conversation plus 8 tokens: a cache hit, one step, must be let through
    t1 = time.time()
    r2 = stream_prompt(follow, 0.0, cache=True)
    fu_wait = r2[0][0] - t1

    dec_thread.join()
    server.stop()

    assert r[-1][1] == len(base) and r2[-1][1] == len(follow), "a prompt did not finish"
    text = log.read_text()
    ev = [l.split("]: ", 1)[-1].strip()[:130] for l in text.splitlines() if "admission" in l]

    assert fu_wait < pms(64) / 1000 + 0.05, f"the cache-hit follow-up was held: {fu_wait:.2f}s\n" + "\n".join(ev)
    assert "admission bypass" in text, "the bypass was not reported"

    # and the decision has to come from the cache state, not from the timings: the follow-up is
    # priced by the tokens it still has to process, and it is the only arrival priced that way
    m_bypass = re.search(r"admission bypass: (\d+) tokens still to process", text)
    assert m_bypass, f"no bypass line with an estimate:\n{text[-1500:]}"
    assert int(m_bypass.group(1)) <= 16, f"the follow-up was not seen as a cache hit: {m_bypass.group(1)} tokens"


def test_admission_bypasses_a_fully_cached_prompt(monkeypatch, tmp_path):
    """A newcomer with nothing left to process is the cheapest arrival there is: take it.

    The floors price a newcomer's *remaining* work, and the cheapest remaining work is zero: a
    request whose tokens are all in the slot's cache, which is routine on a box that runs
    --slot-prompt-similarity 0.05 and keeps handing the same slots back. Zero is not free -- the
    server still evaluates one token to get its logits, and a step is what the floors are priced
    in -- so zero remaining tokens is priced as the one step it really is, and 0.0 is left to
    mean "nothing measured yet".

    Before, the estimate returned 0.0 for zero tokens, the caller's `est_ms > 0.0` test read that
    as the second, and the arrival was held for the whole bound: 21 of them on the production box
    with ADMIT_TG_PCT=50 armed, mean 71 s, max 300 s.

    The probe that matters here is the IDENTICAL re-send, the shape production actually hit; the
    sibling test above stops at base + 8 tokens, one step of work, and never reaches zero. The
    three probes below differ only in how much of the prompt the slot already holds: the first
    warms the slot's cache, the second has nothing left to add, and the third is a cold prompt of
    the same size that has to pay for a whole prefill.
    """
    log = tmp_path / "sched-bypass-cached.log"
    # small batches so the estimates separate by ~125x (a fully cached prompt is one step, a
    # cold 1000-token prompt is 125 of them); the admission constants are priced in tokens
    # through a live rate probe, because the box's single-core speed varies ~60x under load
    # and no fixed millisecond floor keeps the two on opposite sides of the bypass bound
    server.n_batch = 8
    server.n_ubatch = 8
    server.n_ctx = 16384  # the tiny model caps each slot at its 2048-token training context anyway
    # one thread pins the rate; the 1600-token decoder stays inside the model's 2048-token
    # context, so no ctx shift is needed to keep it alive past the probes
    server.n_threads = 1
    server.log_path = str(log)
    server.start()
    rate = measure_rate(server)
    server.stop()

    def ms(n_tokens: int) -> int:
        return int(n_tokens * 1000.0 / rate)

    arm(monkeypatch, ITL_MS=1, ADMIT_TG_PCT=1, ADMIT_PP_PCT=0,
        ADMIT_WAIT_MS=ms(256), ADMIT_BYPASS_MS=ms(32), LOG=64)
    server.start()

    base = make_prompt("Once upon a time there was a little girl named Lily. ", 1000)
    decoder = make_prompt("It was a dark and stormy night in the old castle. ", 32)

    # the decoder takes one slot and stays for seconds; every probe below runs on the other one
    dec_thread = threading.Thread(target=stream_decode, args=(decoder, 1600))
    dec_thread.start()
    time.sleep(0.15)

    # 1. a cold prompt, run to leave its tokens cached in the slot; probe 1's own wait is not
    #    asserted: the live floor is deterministic, so a cold prompt waits like any other
    r = stream_prompt(base, 0.0, cache=True)

    # 2. the identical prompt, token for token: nothing left to process, and it must not wait
    t1 = time.time()
    r2 = stream_prompt(base, 0.0, cache=True)
    cached_wait = r2[0][0] - t1

    dec_thread.join()
    server.stop()

    # the second probe is evaluated on the one token the server has to run for its logits, so
    # the progress stream still reports the whole prompt as processed: the reuse is in the
    # slot's cache, not in the report
    assert r[-1][1] == len(base) and r2[-1][1] == len(base), "a prompt did not finish"
    text = log.read_text()
    ev = [l.split("]: ", 1)[-1].strip()[:130] for l in text.splitlines() if "admission" in l]

    assert cached_wait < ms(32) / 1000 + 0.05, f"the fully cached prompt was held: {cached_wait:.2f}s\n" + "\n".join(ev)

    # and the decision has to come from the remaining work, not from the timings: the arrival
    # with nothing left is priced at zero tokens, and none is held for a floor it cannot break
    bypassed = [int(x) for x in re.findall(r"admission bypass: (\d+) tokens still to process", text)]
    assert bypassed, "the fully cached arrival was not bypassed at all:\n" + "\n".join(ev)
    assert set(bypassed) == {0}, f"the bypass was not priced as zero remaining work: {bypassed}"


def test_admission_hold_does_not_swallow_an_oversize_prompt(monkeypatch, tmp_path):
    """A prompt that is going to be rejected has to be rejected, not parked for the bound.

    The admission hold returns before the prompt fill, and the prompt fill is the only place the
    server rejects a task larger than its slot. So a held oversize task never got its 400: the
    client hung for min(admit_wait, the incumbent's remaining time) and was only then told what
    was wrong with it. The skip now stands aside for a task that is going to be rejected, so the
    error comes back in the iteration the task landed in.

    The status, the error type and the two token counts are what the client acts on, and they
    were right before the fix too -- the same error arrived, just at the far end of the bound.
    The timing is therefore the test here, and the hold line is asserted to be there so the test
    cannot pass by never reaching the hold.
    """
    log = tmp_path / "sched-admit-oversize.log"
    # a 2.5 s bound against a decoder set to 20000 tokens, which cannot run out inside it:
    # held, the request cannot come back before the bound expires. ADMIT_FORCE holds every
    # newcomer while anything is in flight, the deterministic way to reach the hold.
    arm(monkeypatch, ADMIT_TG_PCT=0, ADMIT_PP_PCT=0, ADMIT_FORCE=1, ADMIT_WAIT_MS=2500, LOG=64)
    server.n_ctx = 16384  # each slot caps at the tiny model's 256, which the oversize one exceeds
    server.log_path = str(log)
    server.start()

    dec_prompt = make_prompt("The quick brown fox jumps over the lazy dog. ", 32)
    oversize = make_prompt("Once upon a time there was a little girl named Lily. ", 2200)

    # the decoder takes one slot and stays in flight for the whole bound; the oversize request
    # lands on the other one, so the only thing between it and its 400 is the hold
    dec_thread = threading.Thread(target=stream_decode, args=(dec_prompt, 20000))
    dec_thread.start()
    time.sleep(0.15)

    t0 = time.time()
    res = server.make_request("POST", "/completion", data={
        "prompt": oversize, "n_predict": 1, "stream": False, "ignore_eos": True,
        "cache_prompt": False, "temperature": 0.0, "seed": 42,
    })
    waited = time.time() - t0

    dec_thread.join()
    server.stop()

    text = log.read_text()
    ev = [l.split("]: ", 1)[-1].strip()[:130] for l in text.splitlines() if "admission" in l]

    assert res.status_code == 400, f"the oversize prompt was not rejected: {res.status_code} {res.body}"
    err = res.body["error"]
    assert err["code"] == 400 and err["type"] == "exceed_context_size_error", res.body
    # the error states the two quantities that make it actionable: how long the prompt was, and
    # the slot it did not fit in
    assert err["n_prompt_tokens"] == len(oversize), res.body
    assert err["n_ctx"] < len(oversize), res.body
    # the hold was armed and did fire, so the two paths above really did meet
    assert "admission hold" in text, "the oversize arrival was never held:\n" + "\n".join(ev)
    # and the error did not wait the bound out to arrive
    assert waited < 1.0, f"the rejection waited {waited:.2f}s, the bound is 2.5s:\n" + "\n".join(ev)


def test_state_dump_is_emitted(monkeypatch, tmp_path):
    """The estimate table is observable without arming either knob."""
    log = tmp_path / "sched.log"
    arm(monkeypatch, LOG=20)
    server.log_path = str(log)
    server.start()
    run_overlap()
    server.stop()

    text = log.read_text()
    assert "step sched: steps = " in text
    assert "decode_gap" in text
    # the two arms are reported separately, because they are measured on different axes
    assert "prompt arm (prompt position)" in text
    assert "decode arm (decoder depth)" in text
    assert "admission floors:" in text
    # and a bucket that saw work must report samples for it
    arm_re = re.compile(r"pos (\S+)\s+solo =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\) "
                        r"shared =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\) "
                        r"with_dec =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\)")
    rows = [m for m in (arm_re.search(l) for l in text.splitlines()) if m]
    assert rows, f"no prompt-arm rows in the dump"
    assert any(int(m.group(4)) >= 1 or int(m.group(7)) >= 1 or int(m.group(10)) >= 1 for m in rows), \
        "prompt-arm rows have no samples"


def test_hold_estimate_probes_until_it_can_decide(monkeypatch, tmp_path):
    """The estimate-gated hold must reach a decision instead of probing forever.

    `delta` needs both arms of PROMPT steps, and the no-decoder arm only exists because the
    scheduler takes held probes: a decoder that rides every step leaves no counterfactual.
    Both arms, the make-up price and the resulting delta therefore have to appear in the
    dump. This is the property the production box falsified after 95744 steps -- the arms
    were filed on different axes (the no-decoder arm by a decoder depth it did not have), so
    its samples all landed in the first bucket while the with-decoder arm sat in the bucket
    of the decoder's depth, and no delta was ever formed while the probes kept running.

    NOTE: this test runs on the tiny model, whose slots clamp to a 2048-token training
    context, so both axes land in the shallow bucket here and the axis fix itself is not
    discriminated. The production dump is the evidence for that half (`pos` and `q` buckets
    are now separate axes; see patches/llamacpp/0244).
    """
    log = tmp_path / "sched-estimate.log"
    arm(monkeypatch, HOLD=1, PROBE_EVERY=2, LOG=32)
    server.log_path = str(log)
    # the tiny model clamps its slots to its 2048-token training context, so the prompt and
    # the decoder have to share that: 28 batches of prompt against a 2000-token generation
    server.n_ctx = 4096
    server.start()

    prompt_dec = make_prompt("The quick brown fox jumps over the lazy dog. ", 32)
    prompt_pre = make_prompt("Once upon a time there was a little girl named Lily. ", 28 * N_BATCH)

    parallel_function_calls([
        (stream_decode, (prompt_dec, 2000)),
        (stream_prompt, (prompt_pre, 0.0)),
    ])
    server.stop()

    lines = log.read_text().splitlines()
    devs = collections.defaultdict(list)
    arm_re = re.compile(r"pos (\S+)\s+solo =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\) "
                        r"shared =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\) "
                        r"with_dec =\s+([\d.]+) ms/\s+([\d.]+) tok \(n=(\d+)\)")
    dec = re.compile(r"q (\S+)\s+dec_only =\s+([\d.]+) ms \(n=(\d+)\) k = ([\d.]+) tok")
    delta = re.compile(r"-> delta = (.*?), share ratio = (.*)$")
    for i, l in enumerate(lines):
        m = arm_re.search(l)
        if m:
            dm = delta.search(lines[i + 1]) if i + 1 < len(lines) else None
            devs[m.group(1)].append(("pre", int(m.group(4)), int(m.group(7)), int(m.group(10)),
                                     dm.group(1) if dm else "n/a",
                                     dm.group(2) if dm else "n/a"))
        m = dec.search(l)
        if m:
            devs[m.group(1)].append(("dec", int(m.group(3))))

    assert devs, f"no arm rows in the dump:\n{''.join(lines[-20:])}"
    rows = [t for bucket in devs.values() for t in bucket]

    # the held probes land in the prompt arm's solo/with_dec pair, so both must have samples
    probe_ok = any(t[0] == "pre" and t[1] >= 8 and t[3] >= 8 for t in rows)
    assert probe_ok, f"the two prompt arms never both reached MIN_SAMPLES: {dict(devs)}"

    # and once they have, the dump states a decision rather than refusing to
    decided = any(t[0] == "pre" and t[1] >= 8 and t[3] >= 8 and t[4] != "n/a" for t in rows)
    assert decided, f"the estimate never reached a delta: {dict(devs)}"

    # the decode arm is reported on its own axis, with the make-up step priced
    assert any(t[0] == "dec" and t[1] >= 8 for t in rows), f"no decode arm samples: {dict(devs)}"


def test_state_dump_counts_only_decode_only_steps(monkeypatch, tmp_path):
    """The decode arm's n counts decode-only steps, not every step a decoder advanced.

    t_dec_only and k_tokens are sampled on different sets of steps -- t_dec_only only where the
    batch carried no prompt, k wherever a decoder advanced -- and the gates that read MIN_SAMPLES
    read the first. One counter for both spent the warm-up running mean on samples the
    decode-only estimate never saw, could satisfy the gate with steps that never measured a
    decode-only step (which is what admit_ratio_tg fires on), and reported a count roughly twice
    the truth, so the operator's only view of the identification state was wrong.

    The workload separates the two sets: small batches spread the prompt over many steps and the
    decoder outlives it, so most steps carry a prompt AND a decoder and only the tail is
    decode-only. A decoder advances exactly one token per iteration, so the number of steps the
    prompt could have needed at all bounds the number of decode-only ones from above, and an
    upper bound is the safe direction: on this run every iteration advances the decoder, so the
    pre-fix counter counted all of them and reported n ~= n_steps.
    """
    log = tmp_path / "sched-dec-count.log"
    arm(monkeypatch, LOG=32)
    server.n_batch = 8
    server.n_ubatch = 8
    server.n_ctx = 16384  # each slot caps at the tiny model's 2048, which the prompt below fits
    server.log_path = str(log)
    server.start()

    dec_prompt = make_prompt("The quick brown fox jumps over the lazy dog. ", 32)
    pre = make_prompt("Once upon a time there was a little girl named Lily. ", 2000)

    dec_times, prog = parallel_function_calls([
        (stream_decode, (dec_prompt, N_PREDICT)),
        (stream_prompt, (pre, 0.0)),
    ])
    server.stop()

    assert prog and prog[-1][1] == len(pre), f"the prompt never finished: {prog[-1][1] if prog else None}"
    assert len(dec_times) >= N_PREDICT - 5, f"the decoder stopped early: {len(dec_times)}"

    text = log.read_text()
    dumps = re.findall(r"step sched: steps = (\d+) ", text)
    decs = re.findall(r"q (\S+)\s+dec_only =\s+([\d.]+) ms \(n=(\d+)\)", text)
    assert dumps and decs, f"no decode arm in the dump:\n{text[-1500:]}"
    n_steps = int(dumps[-1])
    n_dec = int(decs[-1][2])

    # even a full batch of its own per iteration is the fewest steps this prompt can have taken,
    # and every remaining step is a candidate decode-only one; one dump interval of slack,
    # because the periodic dump can land inside the tail rather than at its end
    n_pre = len(pre) // server.n_batch
    assert n_dec <= n_steps - n_pre + 32, \
        f"the decode arm counted steps that carried a prompt: n={n_dec} of {n_steps} steps, " \
        f"and at most {n_steps - n_pre} of them had no prompt in the batch"
    # and the count is not vacuous: the decoder outlived the prompt, so there is a tail to count
    assert n_dec >= 1, f"the decoder finished with the prompt: n={n_dec} of {n_steps} steps"


def test_admission_releases_when_the_decoder_turn_ends(monkeypatch, tmp_path):
    """A prompt held for a decoder turn is let through when the turn ends, not after the wait bound.

    The wait bound is a safety cap for a stuck incumbent, not a target: once the decoder's turn
    finishes the box is empty, and the held prompt's remaining wait buys nothing. Holding it for
    the full bound trades the newcomer's time-to-first-token for nothing.
    """
    log = tmp_path / "sched-turn-end.log"
    server.n_ctx = 8192
    server.n_slots = 2
    server.n_threads = 1
    server.log_path = str(log)
    server.start()
    rate = measure_rate(server)
    server.stop()

    # a 4-second turn at the probed rate, capped by the per-slot context; the subject arrives
    # 0.5 s in, so it is held for the turn's remaining time and must release the moment it ends
    turn = min(int(rate * 4.0), 4000)

    arm(monkeypatch, ADMIT_TG_PCT=100, ADMIT_PP_PCT=0, ADMIT_FORCE=1,
        ADMIT_WAIT_MS=15000, ADMIT_BYPASS_MS=0, LOG=64)
    server.start()

    dec_prompt = make_prompt("It was a dark and stormy night in the old castle. ", 32)
    subject = make_prompt("Once upon a time there was a little girl named Lily. ", 512)

    dec_end = {}

    def dec():
        stream_decode(dec_prompt, turn)
        dec_end["t"] = time.time()

    dec_thread = threading.Thread(target=dec)
    dec_thread.start()
    time.sleep(0.5)

    subj_start = time.time()
    r = stream_prompt(subject, 0.0)
    subj_first = r[0][0]
    subj_done = r[-1][1]
    dec_thread.join()
    server.stop()

    assert subj_start < dec_end["t"], "the subject arrived after the turn ended; the hold was vacuous"
    assert subj_done == len(subject), f"the subject never finished: {subj_done}"
    assert subj_first - dec_end["t"] < 1.0, \
        f"the held prompt waited {subj_first - dec_end['t']:.1f}s past the turn end instead of releasing"




def test_admission_admits_when_remainder_beats_prefill(monkeypatch, tmp_path):
    """A held prompt is admitted when the incumbent's remaining work is cheaper than the prefill it costs, not only when the box empties.

    Waiting out the incumbent's full remaining work is the safe default, but once that remaining work is
    cheaper than the newcomer's own prefill, admitting now and sharing the batch beats waiting: the
    newcomer's time-to-first-token drops by the prefill span for a decoder slip the floor already
    tolerates. The remainder is measured, not guessed from a probed rate, so the test holds under load.
    """
    log = tmp_path / "sched-remain.log"
    server.n_ctx = 16384
    server.n_slots = 2
    server.n_threads = 1
    server.log_path = str(log)
    server.start()
    dec_rate = measure_rate(server)
    pp_rate = measure_shared_pp_rate(server)
    server.stop()

    subject_tok = 256  # the full per-slot context: the prefill is the largest the slot allows
    est_s = subject_tok / pp_rate  # the subject's prefill cost, in seconds
    turn = 1  # a single-token turn: the measured remainder stays under the prefill

    arm(monkeypatch, ADMIT_TG_PCT=50, ADMIT_PP_PCT=0, ADMIT_FORCE=1,
        ADMIT_WAIT_MS=15000, ADMIT_BYPASS_MS=0, LOG=64)
    server.start()

    dec_prompt = make_prompt("It was a dark and stormy night in the old castle. ", 32)
    subject = make_prompt("Once upon a time there was a little girl named Lily. ", subject_tok)

    dec_end = {}

    def dec():
        stream_decode(dec_prompt, turn)
        dec_end["t"] = time.time()

    dec_thread = threading.Thread(target=dec)
    dec_thread.start()
    time.sleep(0.001)  # the subject arrives mid-turn

    subj_start = time.time()
    r = stream_prompt(subject, 0.0)
    ttft = r[0][0] - subj_start
    dec_thread.join()
    server.stop()

    remainder_s = dec_end["t"] - subj_start
    assert subj_start < dec_end["t"], "the subject arrived after the turn ended; the hold was vacuous"
    assert remainder_s < est_s, \
        f"the remainder ({remainder_s:.2f}s) is not smaller than the {est_s:.2f}s prefill; the test needs a short incumbent"
    assert r[-1][1] == len(subject), f"the subject never finished: {r[-1][1]}"
    assert ttft < est_s, \
        f"the held prompt waited {ttft:.2f}s; admitting once the {remainder_s:.2f}s remainder is under " \
        f"the {est_s:.2f}s prefill would have been faster"

def test_limitless_turn_prices_its_remainder(monkeypatch, tmp_path):
    """A limitless turn still gets priced, so the guard can release a held prompt before it ends.

    Every production turn is limitless: no client cap and no --n-predict, so n_remaining() is -1
    and the guard had a sentinel (2^20 tokens) in its place. That priced every remainder at
    hours, the "release early when the remainder is cheaper than the newcomer's prefill" rule
    never fired, and a held prompt waited out the incumbent's whole turn (on the box, 98% of
    holds ended exactly when the incumbent's turn did).

    The two phases differ only in whether the turn length has been measured. Unmeasured, there
    is nothing to price the incumbent with, so the subject waits for the bound; once ten short
    turns have been seen, an incumbent already deeper than that owes nothing and the subject is
    let through at once, while the incumbent is still generating.
    """
    log = tmp_path / "sched-limitless.log"
    arm(monkeypatch, ADMIT_TG_PCT=0, ADMIT_PP_PCT=0, ADMIT_FORCE=1, ADMIT_WAIT_MS=400,
        ADMIT_BYPASS_MS=0, LOG=64)
    server.n_threads = 1
    # two slots; the tiny model caps each at its 2048-token training context, which is what a
    # limitless incumbent runs to -- ~1.2 s here, comfortably past the 400 ms bound the
    # unmeasured phase has to wait out
    server.n_ctx = 4096
    server.n_predict = -1  # the server default: a request that sends -1 stays limitless
    server.log_path = str(log)
    def probe(warm_up: int):
        # the prompts need the server up: /tokenize is what makes them exact token ids
        server.start()
        short = make_prompt("The quick brown fox jumps over the lazy dog. ", 16)
        subject = make_prompt("Once upon a time there was a little girl named Lily. ", 256)
        incumbent = make_prompt("It was a dark and stormy night in the old castle. ", 16)
        for _ in range(warm_up):
            stream_decode(short, 12)
        before = log.read_text() if warm_up == 0 else log.read_text()
        dec_end = {}

        def dec():
            stream_decode(incumbent, -1)
            dec_end["t"] = time.time()

        dec_thread = threading.Thread(target=dec)
        dec_thread.start()
        time.sleep(0.05)
        t0 = time.time()
        r = stream_prompt(subject, 0.0)
        ttft = r[0][0] - t0
        dec_thread.join()
        server.stop()
        assert r[-1][1] == len(subject), f"the subject never finished: {r[-1][1]}"
        assert dec_end["t"] > t0 + ttft, "the incumbent's turn ended first; the probe is vacuous"
        return ttft, log.read_text()[len(before):]

    # unmeasured: no turn to price the incumbent with, so the bound is what releases the subject
    t_unmeasured, log_unmeasured = probe(warm_up=0)
    assert "admission hold" in log_unmeasured, "the subject was not held with no turn measured"
    assert t_unmeasured >= 0.30, \
        f"the subject was let through in {t_unmeasured:.2f}s with nothing measured"

    # measured: an incumbent past the measured turn length owes nothing, so the release fires in
    # the iteration the subject arrives -- an order of magnitude under the bound, and long before
    # the incumbent's turn ends
    t_measured, _ = probe(warm_up=10)
    assert t_measured < 0.25, \
        f"the subject waited {t_measured:.2f}s behind an incumbent past its measured turn length"


def test_bfs_leftover_capacity_redistributed(monkeypatch):
    """A short prompt's unused quantum is redistributed to the long prompt,
    so the batch capacity is used efficiently."""
    arm(monkeypatch, LOG=32)
    server.n_batch = 64
    server.n_ctx = 2048
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt_short = make_prompt("Short prompt. ", 10)
    prompt_long = make_prompt("Long prompt. ", 512)
    results = parallel_function_calls([
        (stream_prompt, (prompt_short, 0.0)),
        (stream_prompt, (prompt_long, 0.0)),
    ])
    server.stop()
    # The long prompt should take fewer steps with the redistributed capacity
    # Current behavior: 16 steps (32 tokens per step)
    # Improved behavior: 8 steps (54 tokens in step 1, 64 tokens in steps 2-8)
    long_reports = results[1]
    assert len(long_reports) <= 9, f"long prompt took {len(long_reports)} steps, expected <= 9"

def test_adaptive_batch_size_fits_itl_budget(monkeypatch):
    # the adaptive batch size shrinks n_batch so a step fits the ITL budget: with a
    # tight ITL target the fixed batch overshoots and stalls the decoder, the
    # adaptive batch trades throughput for ITL compliance
    arm(monkeypatch, ITL_MS=100, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive batch test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    # with a 100 ms ITL target and a 64-token batch, the adaptive batch size
    # should keep the step duration at or below the target: the fixed batch
    # (64 tokens) takes ~200 ms on the test model, so without the adaptive
    # cap the ITL would be ~200 ms
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        # allow warm-up: skip the first two gaps
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 300, f"ITL {max_gap:.0f} ms exceeds 3x the 100 ms target"

def test_step_time_itl_target_reduces_itl(monkeypatch):
    # the step-time ITL target uses the measured step time as the effective
    # ITL target when it is below the configured target: with a fast GPU
    # the step time is well under 400 ms, so the ITL should track the
    # step time rather than hover at the target
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Step time ITL test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 300, f"ITL {max_gap:.0f} ms not reduced below the 400 ms target"

def test_composition_aware_batch_cost(monkeypatch):
    # the batch cap uses a composition-weighted ms/token: a mostly-prompt
    # batch has a much lower ms/token than the decode-heavy average, so the
    # cap should allow a larger batch and the prefill throughput should be
    # higher than a uniform estimate would permit
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Composition aware test. ", 512)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        total_time = reports[-1][0] - reports[0][0]
        if total_time > 0:
            throughput = reports[-1][1] / total_time
            assert throughput > 5000, f"prefill throughput {throughput:.0f} t/s too low"

def test_per_decoder_itl_target(monkeypatch):
    # the per-decoder ITL target divides the configured ITL by the number of
    # active decoders: with 2 decoders and a 400 ms target the effective
    # target is 200 ms, so the per-decoder ITL should be ~400 ms (2 x 200)
    # rather than 800 ms (2 x 400)
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Per decoder ITL test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 600, f"per-decoder ITL {max_gap:.0f} ms exceeds 600 ms"

def test_dynamic_max_share_scales_with_decoders(monkeypatch):
    # the dynamic max_share divides the prompt budget by the decoder count:
    # with 2 decoders and a 33 pct ceiling the effective max_share is 33
    # (100/3), so the prompt budget is unchanged. the test verifies the
    # prompt still makes progress under the dynamic budget
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Dynamic max share test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        total_time = reports[-1][0] - reports[0][0]
        if total_time > 0:
            throughput = reports[-1][1] / total_time
            assert throughput > 1000, f"prompt throughput {throughput:.0f} t/s too low"

def test_load_scaled_admission_threshold(monkeypatch):
    # the load-scaled admission threshold tightens as the decoder count grows:
    # with 2 decoders and a 50 pct floor the effective floor is 75 pct
    # (50 + 50*2/4), so the scheduler is more conservative about admitting
    # new prompts. the test verifies the prompt still makes progress under
    # the tightened threshold
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Load scaled admission test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        total_time = reports[-1][0] - reports[0][0]
        if total_time > 0:
            throughput = reports[-1][1] / total_time
            assert throughput > 500, f"prompt throughput {throughput:.0f} t/s too low"

def test_per_decoder_injection_gate(monkeypatch):
    # the per-decoder injection gate divides the ITL target by the decoder
    # count: with 2 decoders and a 400 ms target the effective injection
    # threshold is 200 ms, so the scheduler injects decode-only steps more
    # frequently. the test verifies the per-decoder ITL stays within the
    # configured target under the tighter gate
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Per decoder injection test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 600, f"per-decoder ITL {max_gap:.0f} ms exceeds 600 ms"
def test_adaptive_itl_target_relaxes_under_load(monkeypatch):
    # the adaptive ITL target scales up with decoder count: at 4 decoders the
    # per-decoder target is itl_ms * 8 / 16 = 200 ms, not itl_ms / 4 = 100 ms.
    # this keeps the GPU busier on prompts while still bounding per-decoder ITL.
    # the test verifies the per-decoder ITL stays within the relaxed target
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive ITL target test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            # relaxed target: 400 * 8 / 16 = 200 ms per decoder, allow 2x headroom
            assert max_gap < 500, f"adaptive ITL {max_gap:.0f} ms exceeds 500 ms relaxed bound"

def test_itl_proportional_max_share_throttles_prompt(monkeypatch):
    # the adaptive max_share scales with the live ITL gap: when the decoder gap
    # is near the target, prompt share drops toward zero; when the gap is small,
    # prompt share approaches the configured max. this keeps the GPU on prompts
    # only when the decoders can absorb the interruption.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=50, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("ITL proportional max share test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"ITL-proportional max_share let ITL reach {max_gap:.0f} ms"
def test_itl_aware_admission_floor_holds_under_pressure(monkeypatch):
    # the ITL-aware admission floor raises the bar when the decoder gap is
    # near the target: a second prompt that would have been admitted at the
    # static floor is held when the gap is hot. the test verifies that the
    # per-decoder ITL stays bounded under 4-way concurrency.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("ITL aware admission floor test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"ITL-aware admission floor let ITL reach {max_gap:.0f} ms"
def test_predictive_injection_uses_ewma_decode_time(monkeypatch):
    # the injection gate predicts the next decode step's duration via EWMA
    # rather than reacting only to the current gap: a decoder whose gap is
    # still under target but whose decode steps are trending long gets an
    # injected step before the gap crosses the line. the test verifies the
    # per-decoder ITL stays bounded under 2-way concurrency with the
    # predictive gate active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 2
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Predictive injection test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"predictive injection let ITL reach {max_gap:.0f} ms"
def test_adaptive_streak_limit_relaxes_under_load(monkeypatch):
    # the streak limit relaxes as the decoder count grows: at 4 decoders the
    # prompt can make less progress between injection episodes, so the cap
    # rises to let the decoders get their steps. the test verifies the
    # per-decoder ITL stays bounded under 4-way concurrency with the
    # adaptive streak limit active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive streak limit test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"adaptive streak limit let ITL reach {max_gap:.0f} ms"
def test_adaptive_chunk_size_shrinks_under_load(monkeypatch):
    # the prompt chunk shrinks as the decoder count grows: at 4 decoders the
    # chunk is half the base size, so each prompt step takes less time and the
    # decoder's ITL gap is shorter. the test verifies the per-decoder ITL stays
    # bounded under 4-way concurrency with the adaptive chunk size active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive chunk size test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"adaptive chunk size let ITL reach {max_gap:.0f} ms"
def test_adaptive_probe_interval_shrinks_under_load(monkeypatch):
    # the probe interval shrinks as the decoder count grows: at 4 decoders the
    # probe fires twice as often, so the rate estimate is fresher and the
    # injection gate makes better decisions. the test verifies the per-decoder
    # ITL stays bounded under 4-way concurrency with the adaptive probe active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive probe interval test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"adaptive probe interval let ITL reach {max_gap:.0f} ms"
def test_adaptive_hold_margin_tightens_under_load(monkeypatch):
    # the hold margin grows as the decoder count grows: at 4 decoders the
    # margin is 50 pct larger, so the scheduler holds prompts more aggressively
    # and the decoder's ITL gap stays tighter. the test verifies the per-decoder
    # ITL stays bounded under 4-way concurrency with the adaptive margin active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive hold margin test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"adaptive hold margin let ITL reach {max_gap:.0f} ms"
def test_hybrid_step_prediction_tracks_transient_load(monkeypatch):
    # the prediction blends the EWMA step time with the last actual step, so
    # transient load shifts are tracked faster than the pure EWMA. the test
    # verifies the per-decoder ITL stays bounded under 4-way concurrency with
    # the hybrid prediction active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hybrid step prediction test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hybrid prediction let ITL reach {max_gap:.0f} ms"
def test_adaptive_step_blend_tracks_volatility(monkeypatch):
    # the prediction blends the EWMA step time with the last actual step, and the
    # blend weights adapt to the step-time volatility: when the last step deviates
    # from the EWMA, the blend weights the last step more heavily, so the
    # prediction tracks transient load shifts faster than a fixed-weight blend.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Adaptive step blend test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"adaptive blend let ITL reach {max_gap:.0f} ms"
def test_hyperbolic_itl_relaxation_tracks_load(monkeypatch):
    # the per-decoder ITL target relaxes hyperbolically as the decoder count
    # grows: at 1 decoder the target is 0.8x the base, at 4 decoders 0.5x,
    # at 8 decoders 0.33x. the test verifies the per-decoder ITL stays
    # bounded under 4-way concurrency with the hyperbolic relaxation active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hyperbolic ITL relaxation test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hyperbolic relaxation let ITL reach {max_gap:.0f} ms"
def test_quadratic_max_share_shrinkage(monkeypatch):
    # the max share of step time the prompt pays shrinks quadratically as the
    # decoder's gap grows: at half the ITL target the prompt keeps 75% of its
    # max share, at the full target it keeps 0%. the test verifies the
    # per-decoder ITL stays bounded under 4-way concurrency with the
    # quadratic shrinkage active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Quadratic max share shrinkage test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"quadratic shrinkage let ITL reach {max_gap:.0f} ms"
def test_prompt_urgency_lets_nearly_done_prompts_finish(monkeypatch):
    # when a prompt has few tokens left, the scheduler reduces its injection
    # share so the prompt finishes faster instead of being stretched out by
    # decode-only steps. the test verifies that a short prompt (64 tokens)
    # completes without the ITL degrading under 4-way concurrency.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Prompt urgency test. ", 64)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"prompt urgency let ITL reach {max_gap:.0f} ms"
def test_hyperbolic_streak_limit_relaxes_under_load(monkeypatch):
    # the streak cap relaxes hyperbolically as the decoder count grows: more
    # aggressive at low decoder counts (where the prompt owns most of the
    # batch) and more conservative at high counts (where the decoders
    # dominate). the test verifies that the per-decoder ITL stays bounded
    # under 4-way concurrency with the hyperbolic relaxation active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hyperbolic streak limit test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hyperbolic streak limit let ITL reach {max_gap:.0f} ms"
def test_hyperbolic_chunk_scale_shrinks_under_load(monkeypatch):
    # the chunk scale shrinks hyperbolically as the decoder count grows: more
    # conservative at high decoder counts (keeps the chunk larger so the
    # prompt makes progress) and more aggressive at low counts (smaller
    # chunks keep the ITL gap short). the test verifies that the per-decoder
    # ITL stays bounded under 4-way concurrency with the hyperbolic scale.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hyperbolic chunk scale test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hyperbolic chunk scale let ITL reach {max_gap:.0f} ms"
def test_hyperbolic_probe_interval_shrinks_under_load(monkeypatch):
    # the probe interval shrinks hyperbolically as the decoder count grows:
    # more aggressive at low decoder counts (fresher rate estimates) and more
    # conservative at high counts (the estimate is already stale). the test
    # verifies that the per-decoder ITL stays bounded under 4-way concurrency
    # with the hyperbolic probe interval active.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hyperbolic probe interval test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hyperbolic probe interval let ITL reach {max_gap:.0f} ms"
def test_hyperbolic_hold_margin_tightens_under_load(monkeypatch):
    # the hold margin tightens hyperbolically as the decoder count grows:
    # more aggressive at low decoder counts (the estimate is less reliable
    # with fewer samples) and more conservative at high counts (the
    # estimate is more reliable). the test verifies that the per-decoder
    # ITL stays bounded under 4-way concurrency with the hyperbolic margin.
    arm(monkeypatch, ITL_MS=400, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_slots = 4
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Hyperbolic hold margin test. ", 256)
    results = parallel_function_calls([
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
        (stream_prompt, (prompt, 0.0)),
    ])
    server.stop()
    for reports in results:
        if len(reports) < 2:
            continue
        gaps = [reports[i + 1][0] - reports[i][0] for i in range(len(reports) - 1)]
        steady = gaps[2:] if len(gaps) > 2 else gaps
        if steady:
            max_gap = max(steady) * 1000.0
            assert max_gap < 500, f"hyperbolic hold margin let ITL reach {max_gap:.0f} ms"
def test_batch_cap_is_not_pinned_below_its_floor(monkeypatch):
    # a step costs a fixed part plus a marginal part per token, and mpc is an
    # average: on a batch at the floor the fixed part dominates, so n_fit stays
    # below the floor and clamping there held the batch at 32 tokens/step for
    # the life of the process -- every step re-measured the same small batch.
    # a target no batch size can meet must lift the cap, not pin the batch.
    arm(monkeypatch, ITL_MS=5, MAX_SHARE=33, ADMIT_TG_PCT=50, ADMIT_PP_PCT=100,
        ADMIT_WAIT_MS=240000, ADMIT_BYPASS_MS=3000)
    server.n_ctx = 4096
    server.n_slots = 2
    server.n_batch = 128
    server.n_ubatch = 128
    server.n_predict = 1
    server.start()
    prompt = make_prompt("Batch cap floor test. ", 1792)
    results = parallel_function_calls([(stream_prompt, (prompt, 0.0))])
    server.stop()
    reports = results[0]
    # one prompt takes the whole batch: 1792 tokens at 128 per step is 14 steps,
    # while the cap can only bite after 8 samples, after which a batch pinned at
    # the 32-token floor needs 8 + (1792 - 1024)/32 = 32
    assert len(reports) <= 20, \
        f"the cap pinned the prompt at {1792 / len(reports):.0f} tokens/step"
